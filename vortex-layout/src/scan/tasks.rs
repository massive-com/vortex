// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use bit_vec::BitVec;
use futures::FutureExt;
use futures::channel::oneshot;
use futures::future::BoxFuture;
use futures::future::ok;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scan::selection::Selection;

use crate::LayoutReader;
use crate::scan::filter::FilterExpr;

pub type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

/// Atomically reserve up to `available` rows from the shared `remaining` counter.
///
/// Returns the number of rows actually reserved (which may be less than `available` if
/// the counter ran out). Uses a CAS loop so that concurrent splits cannot underflow the
/// counter or oversubscribe past the requested limit.
///
/// The reservation is bounded by `available` so the returned value always fits back into
/// the same type.
fn reserve_up_to(remaining: &AtomicU64, available: usize) -> usize {
    // Saturate the available rows to u64 so we can CAS on the shared atomic. Since the
    // reservation is bounded by `available`, the returned value is bounded by it too,
    // and we can cast back to usize without losing precision.
    let available_u64 = u64::try_from(available).unwrap_or(u64::MAX);
    let mut current = remaining.load(Ordering::Acquire);
    let reserved_u64 = loop {
        if current == 0 {
            return 0;
        }
        let take = current.min(available_u64);
        match remaining.compare_exchange_weak(
            current,
            current - take,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => break take,
            Err(observed) => current = observed,
        }
    };
    // `reserved_u64 <= available_u64 <= available`, so it fits in usize.
    usize::try_from(reserved_u64).unwrap_or(available)
}

/// How a split task applies the row limit of the scan.
pub enum SplitLimit {
    /// All splits reserve rows from one shared counter. The split that reserves first gets the
    /// rows.
    ///
    /// Use it only where this order does not matter or is the scan order: for unordered scans,
    /// and for scans without a filter, where each split reserves when its task is built.
    Shared(Arc<AtomicU64>),
    /// Each split reserves its rows after the previous split in scan order.
    ///
    /// The split receives the remaining row count from the previous split, keeps up to that many
    /// rows, and sends the rest to the next split. The filters of all splits still run
    /// concurrently. Only the reservation waits for the previous split.
    Chained {
        /// The remaining row count after the previous split.
        previous: oneshot::Receiver<u64>,
        /// The remaining row count after this split.
        next: oneshot::Sender<u64>,
    },
}

impl SplitLimit {
    /// Reserves rows for this split and limits `mask` to the reserved rows.
    ///
    /// If `keep_last` is true, the split keeps its last selected rows. A reversed scan needs this,
    /// because it reads a split in natural order and reverses the rows afterwards.
    async fn apply(self, mask: Mask, keep_last: bool) -> Mask {
        let true_count = mask.true_count();
        let reserved = match self {
            SplitLimit::Shared(remaining) => reserve_up_to(&remaining, true_count),
            SplitLimit::Chained { previous, next } => {
                // The sender is dropped only if the previous split failed. Its error ends the
                // ordered stream before this split's rows.
                let remaining = previous.await.unwrap_or(0);
                let reserved = usize::try_from(remaining).map_or(true_count, |r| r.min(true_count));
                let reserved_u64 = u64::try_from(reserved).unwrap_or(remaining);
                // The receiver is dropped if the consumer stopped the scan. No split needs the
                // count then.
                next.send(remaining.saturating_sub(reserved_u64)).ok();
                reserved
            }
        };
        limit_mask(mask, reserved, keep_last)
    }
}

/// Keeps the first (or with `keep_last`, the last) `reserved` selected rows of `mask`.
fn limit_mask(mask: Mask, reserved: usize, keep_last: bool) -> Mask {
    if reserved == 0 {
        Mask::new_false(mask.len())
    } else if keep_last {
        mask.limit_last(reserved)
    } else {
        mask.limit(reserved)
    }
}

/// Logic for executing a single split reading task.
/// N.B. read_mask should be evaluated against all_false() before calling this
/// method to avoid creating an empty TaskFuture.
///
/// # Task execution flow
///
/// First, the task's row range (split) is intersected with the global file row-range requested,
/// if any.
///
/// The intersected row range is then further reduced via expression-based pruning. After pruning
/// has eliminated more blocks, the full filter is executed over the remainder of the split.
///
/// This mask is then provided to the reader to perform a filtered projection over the split data,
/// finally mapping the Vortex columnar record batches into some result type `A`.
pub fn split_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    split: Range<u64>,
    limit: Option<SplitLimit>,
) -> VortexResult<TaskFuture<Option<A>>> {
    let read_mask = ctx.selection.row_mask(&split);
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();
    let keep_last = ctx.reversed;
    if row_mask.all_false() {
        // A chained limit must still pass the remaining row count to the next split.
        let len = row_mask.len();
        return Ok(async move {
            if let Some(limit) = limit {
                limit.apply(Mask::new_false(len), keep_last).await;
            }
            Ok(None)
        }
        .boxed());
    }

    // Early exit: if a shared limit is set and already exhausted, skip IO entirely.
    if let Some(SplitLimit::Shared(remaining)) = limit.as_ref()
        && remaining.load(Ordering::Acquire) == 0
    {
        return Ok(ok(None).boxed());
    }

    let filter_mask = match ctx.filter.as_ref() {
        // No filter == immediate mask
        None => match limit {
            Some(SplitLimit::Shared(remaining)) => {
                let true_count = row_mask.true_count();
                let reserved = reserve_up_to(&remaining, true_count);
                MaskFuture::ready(limit_mask(row_mask, reserved, keep_last))
            }
            Some(chained @ SplitLimit::Chained { .. }) => {
                MaskFuture::new(row_mask.len(), async move {
                    Ok(chained.apply(row_mask, keep_last).await)
                })
            }
            None => MaskFuture::ready(row_mask),
        },
        Some(filter) => {
            // NOTE: it's very important that the pruning and filter evaluations are built OUTSIDE
            // the future. Registering these row ranges eagerly is a hint to the IO system that
            // we want to start prefetching the IO for this split.
            let reader = Arc::clone(&ctx.reader);
            let filter = Arc::clone(filter);
            let row_range = row_range.clone();

            MaskFuture::new(row_mask.len(), async move {
                let filtered = async move {
                    let mut mask = row_mask;
                    let mut dynamic_versions = vec![None; filter.conjuncts().len()];

                    // TODO(ngates): we could use FuturedUnordered to intersect the masks in parallel.
                    for (idx, conjunct) in filter.conjuncts().iter().enumerate() {
                        if mask.all_false() {
                            return Ok(mask);
                        }

                        // Store the latest version of the dynamic expression prior to pruning.
                        // We will re-run the pruning later if the version has changed in the meantime.
                        dynamic_versions[idx] = filter.dynamic_updates(idx).map(|du| du.version());

                        let conjunct_mask = reader
                            .pruning_evaluation(&row_range, conjunct, mask.clone())?
                            .await?;
                        mask = mask.bitand(&conjunct_mask);
                    }

                    // Now we loop through the conjuncts in the preferred order and evaluate them.
                    let mut remaining = BitVec::from_elem(filter.conjuncts().len(), true);
                    while let Some(idx) = filter.next_conjunct(&remaining) {
                        remaining.set(idx, false);
                        if mask.all_false() {
                            return Ok(mask);
                        }

                        let conjunct = &filter.conjuncts()[idx];

                        // If the dynamic expression has changed since pruning, re-run the pruning.
                        // Store the dynamic update once to avoid TOCTOU race condition
                        let current_version = filter.dynamic_updates(idx).map(|du| du.version());
                        if let Some(dv) = current_version
                            && dynamic_versions[idx].is_none_or(|v| v < dv)
                        {
                            // The dynamic expression has been updated, re-run the pruning.
                            dynamic_versions[idx] = Some(dv);
                            let conjunct_mask = reader
                                .pruning_evaluation(&row_range, conjunct, mask.clone())?
                                .await?;
                            mask = mask.bitand(&conjunct_mask);
                        }
                        if mask.all_false() {
                            return Ok(mask);
                        }

                        let input_true_count = mask.true_count();
                        let conjunct_mask = reader
                            .filter_evaluation(&row_range, conjunct, MaskFuture::ready(mask))?
                            .await?;
                        filter.report_selectivity(
                            idx,
                            conditional_selectivity(input_true_count, conjunct_mask.true_count()),
                        );

                        // Filter evaluations return a mask already intersected with the input mask.
                        mask = conjunct_mask;
                    }

                    VortexResult::Ok(mask)
                };
                let mask = filtered.await?;

                // Apply the limit after the filter is fully evaluated, so that the split only
                // reserves the rows it emits.
                match limit {
                    Some(limit) => Ok(limit.apply(mask, keep_last).await),
                    None => Ok(mask),
                }
            })
        }
    };

    // Step 4: execute the projection, only at the mask for rows which match the filter
    let projection_future =
        ctx.reader
            .projection_evaluation(&row_range, &ctx.projection, filter_mask.clone())?;

    let mapper = Arc::clone(&ctx.mapper);
    let array_fut = async move {
        let mask = filter_mask.await?;
        if mask.all_false() {
            return Ok(None);
        }

        let array = projection_future.await?;
        mapper(array).map(Some)
    };

    Ok(array_fut.boxed())
}

fn conditional_selectivity(input_true_count: usize, output_true_count: usize) -> f64 {
    debug_assert!(input_true_count > 0);
    debug_assert!(output_true_count <= input_true_count);
    output_true_count as f64 / input_true_count as f64
}

/// Information needed to execute a single split task.
pub struct TaskContext<A> {
    /// A row selection to apply.
    pub selection: Selection,
    /// The shared filter expression.
    pub filter: Option<Arc<FilterExpr>>,
    /// The layout reader.
    pub reader: Arc<dyn LayoutReader>,
    /// The projection expression to apply to gather the scanned rows.
    pub projection: BoundExpression,
    /// Function that maps into an A.
    pub mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    /// Whether the scan emits the rows of each split in reverse order.
    pub reversed: bool,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;
    use std::thread;

    use super::conditional_selectivity;
    use super::reserve_up_to;

    #[test]
    fn selectivity_is_relative_to_the_input_mask() {
        assert_eq!(conditional_selectivity(20, 5), 0.25);
    }

    #[test]
    fn reserve_up_to_drains_counter() {
        let remaining = AtomicU64::new(10);
        assert_eq!(reserve_up_to(&remaining, 4), 4);
        assert_eq!(remaining.load(Ordering::Acquire), 6);
        assert_eq!(reserve_up_to(&remaining, 10), 6);
        assert_eq!(remaining.load(Ordering::Acquire), 0);
        assert_eq!(reserve_up_to(&remaining, 10), 0);
    }

    #[test]
    fn reserve_up_to_concurrent_never_oversubscribes() {
        // Even under heavy contention the CAS loop must never let the counter underflow or
        // hand out more than the initial budget across all racing threads.
        const LIMIT: u64 = 1024;
        const THREADS: usize = 32;
        const PER_THREAD_REQUEST: usize = 64;

        let remaining = Arc::new(AtomicU64::new(LIMIT));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let remaining = Arc::clone(&remaining);
                thread::spawn(move || {
                    let mut total: usize = 0;
                    while total < PER_THREAD_REQUEST {
                        let take = reserve_up_to(&remaining, PER_THREAD_REQUEST - total);
                        if take == 0 {
                            break;
                        }
                        total += take;
                    }
                    total
                })
            })
            .collect();

        let granted: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        let granted_u64 = u64::try_from(granted).unwrap();
        assert!(
            granted_u64 <= LIMIT,
            "granted={granted_u64} exceeded limit={LIMIT}"
        );
        assert_eq!(granted_u64 + remaining.load(Ordering::Acquire), LIMIT);
    }
}
