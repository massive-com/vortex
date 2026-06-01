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
use futures::future::BoxFuture;
use futures::future::ok;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::expr::Expression;
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

/// Logic for executing a single split reading task.
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
    limit: Option<Arc<AtomicU64>>,
) -> VortexResult<TaskFuture<Option<A>>> {
    // Apply the selection to calculate a read mask
    let read_mask = ctx.selection.row_mask(&split);
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();
    if row_mask.all_false() {
        return Ok(ok(None).boxed());
    }

    // Early exit: if a shared limit is set and already exhausted, skip IO entirely.
    if let Some(remaining) = limit.as_ref()
        && remaining.load(Ordering::Acquire) == 0
    {
        return Ok(ok(None).boxed());
    }

    let filter_mask = match ctx.filter.as_ref() {
        // No filter == immediate mask
        None => {
            let row_mask = match limit.as_ref() {
                Some(remaining) => {
                    let true_count = row_mask.true_count();
                    let reserved = reserve_up_to(remaining, true_count);
                    if reserved == 0 {
                        Mask::new_false(row_mask.len())
                    } else {
                        row_mask.limit(reserved)
                    }
                }
                None => row_mask,
            };

            MaskFuture::ready(row_mask)
        }
        Some(filter) => {
            // NOTE: it's very important that the pruning and filter evaluations are built OUTSIDE
            // the future. Registering these row ranges eagerly is a hint to the IO system that
            // we want to start prefetching the IO for this split.
            let reader = Arc::clone(&ctx.reader);
            let filter = Arc::clone(filter);
            let row_range = row_range.clone();

            MaskFuture::new(row_mask.len(), async move {
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

                    let conjunct_mask = reader
                        .filter_evaluation(&row_range, conjunct, MaskFuture::ready(mask))?
                        .await?;
                    filter.report_selectivity(idx, conjunct_mask.density());

                    // Filter evaluations return a mask already intersected with the input mask.
                    mask = conjunct_mask;
                }

                // Apply the shared limit. We do this after the filter is fully evaluated so we
                // only reserve from the global counter what this split would actually emit.
                if let Some(remaining) = limit.as_ref() {
                    let true_count = mask.true_count();
                    let reserved = reserve_up_to(remaining, true_count);
                    mask = if reserved == 0 {
                        Mask::new_false(mask.len())
                    } else if reserved < true_count {
                        mask.limit(reserved)
                    } else {
                        mask
                    };
                }

                Ok(mask)
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

/// Information needed to execute a single split task.
pub struct TaskContext<A> {
    /// A row selection to apply.
    pub selection: Selection,
    /// The shared filter expression.
    pub filter: Option<Arc<FilterExpr>>,
    /// The layout reader.
    pub reader: Arc<dyn LayoutReader>,
    /// The projection expression to apply to gather the scanned rows.
    pub projection: Expression,
    /// Function that maps into an A.
    pub mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;
    use std::thread;

    use super::reserve_up_to;

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
                let remaining = remaining.clone();
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
