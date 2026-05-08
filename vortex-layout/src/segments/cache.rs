// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::FutureExt;
use moka::future::Cache;
use moka::future::CacheBuilder;
use moka::policy::EvictionPolicy;
use rustc_hash::FxBuildHasher;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_metrics::Counter;
use vortex_metrics::Label;
use vortex_metrics::MetricBuilder;
use vortex_metrics::MetricsRegistry;
use vortex_utils::aliases::dash_map::DashMap;

use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// A cache for storing and retrieving individual segment data.
#[async_trait]
pub trait SegmentCache: Send + Sync {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>>;
    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()>;
}

/// Shared, type-erased reference to a [`SegmentCache`].
///
/// This is the form used at almost every API boundary that hands off a [`SegmentCache`]
/// (builder outputs, file open options, source adapters). The alias exists primarily so
/// that IDE "find references" can locate every shared cache hand-off without matching
/// every `Arc<dyn ...>` in the codebase.
pub type SharedSegmentCache = Arc<dyn SegmentCache>;

#[async_trait]
impl<C: SegmentCache + ?Sized> SegmentCache for Arc<C> {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        (**self).get(id).await
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        (**self).put(id, buffer).await
    }
}

pub struct NoOpSegmentCache;

#[async_trait]
impl SegmentCache for NoOpSegmentCache {
    async fn get(&self, _id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(None)
    }

    async fn put(&self, _id: SegmentId, _buffer: ByteBuffer) -> VortexResult<()> {
        Ok(())
    }
}

/// A [`SegmentCache`] based around an in-memory Moka cache.
pub struct MokaSegmentCache(Cache<SegmentId, ByteBuffer, FxBuildHasher>);

impl MokaSegmentCache {
    pub fn new(max_capacity_bytes: u64) -> Self {
        Self(
            CacheBuilder::new(max_capacity_bytes)
                .name("vortex-segment-cache")
                // Weight each segment by the number of bytes in the buffer.
                .weigher(|_, buffer: &ByteBuffer| {
                    u32::try_from(buffer.len().min(u32::MAX as usize)).vortex_expect("must fit")
                })
                // We configure LFU (vs LRU) since the cache is mostly used when re-reading the
                // same file - it is _not_ used when reading the same segments during a single
                // scan.
                .eviction_policy(EvictionPolicy::tiny_lfu())
                .build_with_hasher(FxBuildHasher),
        )
    }
}

#[async_trait]
impl SegmentCache for MokaSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.0.get(&id).await)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.0.insert(id, buffer).await;
        Ok(())
    }
}

/// Wrapper for [`SegmentCache`] that tracks its hit rate.
pub struct InstrumentedSegmentCache<C> {
    segment_cache: C,

    hits: Counter,
    misses: Counter,
    stores: Counter,
}

impl<C: SegmentCache> InstrumentedSegmentCache<C> {
    pub fn new(
        segment_cache: C,
        metrics_registry: &dyn MetricsRegistry,
        labels: Vec<Label>,
    ) -> Self {
        Self {
            segment_cache,
            hits: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("vortex.file.segments.cache.hits"),
            misses: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("vortex.file.segments.cache.misses"),
            stores: MetricBuilder::new(metrics_registry)
                .add_labels(labels)
                .counter("vortex.file.segments.cache.stores"),
        }
    }
}

#[async_trait]
impl<C: SegmentCache> SegmentCache for InstrumentedSegmentCache<C> {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        let result = self.segment_cache.get(id).await?;
        if result.is_some() {
            self.hits.add(1);
        } else {
            self.misses.add(1);
        }
        Ok(result)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.segment_cache.put(id, buffer).await?;
        self.stores.add(1);
        Ok(())
    }
}

/// Decorator [`SegmentCacheBuilder`] that wraps each per-file cache with an
/// [`InstrumentedSegmentCache`] for hit/miss/store metrics.
///
/// # Example
///
/// ```ignore
/// let cache = Arc::new(InstrumentedSegmentCacheBuilder::new(
///     NamespacedMokaSegmentCacheBuilder::new(2 << 30),
///     metrics_registry,
///     vec![],
/// ));
/// ```
pub struct InstrumentedSegmentCacheBuilder<B> {
    inner: B,
    metrics_registry: Arc<dyn MetricsRegistry>,
    labels: Vec<Label>,
}

impl<B: SegmentCacheBuilder> InstrumentedSegmentCacheBuilder<B> {
    /// Wrap `inner` so each per-file cache it produces is instrumented with the given
    /// metrics registry and labels.
    pub fn new(inner: B, metrics_registry: Arc<dyn MetricsRegistry>, labels: Vec<Label>) -> Self {
        Self {
            inner,
            metrics_registry,
            labels,
        }
    }
}

impl<B: SegmentCacheBuilder> SegmentCacheBuilder for InstrumentedSegmentCacheBuilder<B> {
    fn cache_for(&self, file: &FileIdentity) -> SharedSegmentCache {
        Arc::new(InstrumentedSegmentCache::new(
            self.inner.cache_for(file),
            &*self.metrics_registry,
            self.labels.clone(),
        ))
    }
}

pub struct SegmentCacheSourceAdapter {
    cache: SharedSegmentCache,
    source: Arc<dyn SegmentSource>,
}

impl SegmentCacheSourceAdapter {
    pub fn new(cache: SharedSegmentCache, source: Arc<dyn SegmentSource>) -> Self {
        Self { cache, source }
    }
}

impl SegmentSource for SegmentCacheSourceAdapter {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let cache = Arc::clone(&self.cache);
        let delegate = self.source.request(id);

        async move {
            if let Ok(Some(segment)) = cache.get(id).await {
                tracing::debug!("Resolved segment {} from cache", id);
                return Ok(BufferHandle::new_host(segment));
            }
            let result = delegate.await?;
            // Cache only CPU buffers; device buffers are not cached.
            if let Some(buffer) = result.as_host_opt()
                && let Err(e) = cache.put(id, buffer.clone()).await
            {
                tracing::warn!("Failed to store segment {} in cache: {}", id, e);
            }
            Ok(result)
        }
        .boxed()
    }
}

/// Identity of an opened Vortex file, used to scope cross-file [`SegmentCache`] entries.
///
/// Two files compare equal only if both [`path`](FileIdentity::path) and
/// [`version`](FileIdentity::version) match. An overwrite at the same path produces a new
/// `FileVersion`, so old cache entries become unreachable rather than serving stale data.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct FileIdentity {
    /// Logical location of the file (path, URI, etc.).
    pub path: Arc<str>,
    /// Content version. Use [`FileVersion::Etag`] when an etag is available; otherwise
    /// fall back to [`FileVersion::SizeMtime`].
    pub version: FileVersion,
}

/// A content version for a [`FileIdentity`].
///
/// Every file has at least a size and modification time (whether on a local filesystem
/// or in an object store), so a version is always available. Use [`FileVersion::Etag`]
/// when the storage layer provides one — it is the most reliable signal of content
/// change — and [`FileVersion::SizeMtime`] otherwise.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum FileVersion {
    /// Content-derived tag from the storage layer (S3/GCS/Azure etag, etc.).
    Etag(Arc<str>),
    /// Fallback when no etag is available: file size in bytes and modification time
    /// as a Unix timestamp in seconds.
    SizeMtime(u64, i64),
}

/// Hands out a per-file [`SegmentCache`] for a given [`FileIdentity`].
///
/// This is the user-facing trait for configuring cross-file segment caching. The builder
/// owns the shared resources (e.g. a global Moka cache) and decides how to scope per-file
/// keys to avoid collisions between files.
pub trait SegmentCacheBuilder: Send + Sync {
    /// Return a [`SegmentCache`] scoped to `file`.
    fn cache_for(&self, file: &FileIdentity) -> SharedSegmentCache;
}

/// A [`SegmentCacheBuilder`] that returns [`NoOpSegmentCache`] for every file.
pub struct NoOpSegmentCacheBuilder;

impl SegmentCacheBuilder for NoOpSegmentCacheBuilder {
    fn cache_for(&self, _file: &FileIdentity) -> SharedSegmentCache {
        Arc::new(NoOpSegmentCache)
    }
}

/// A [`SegmentCacheBuilder`] backed by a single shared Moka cache, with per-file
/// namespacing so that segment IDs from different files never alias.
///
/// Each unique [`FileIdentity`] is assigned a stable `u32` file ID on first use, and
/// the underlying Moka cache is keyed on `(file_id, segment_id)`. Cross-query reuse is
/// enabled for repeated reads of the same file; overwrites change the [`FileVersion`]
/// and therefore produce a fresh file ID, which is the desired behavior.
pub struct NamespacedMokaSegmentCacheBuilder {
    inner: Arc<NamespacedMokaInner>,
}

struct NamespacedMokaInner {
    cache: Cache<(u32, u32), ByteBuffer, FxBuildHasher>,
    file_ids: DashMap<FileIdentity, u32>,
    next_file_id: AtomicU32,
}

impl NamespacedMokaSegmentCacheBuilder {
    /// Create a new builder backed by a Moka cache of the given total capacity in bytes.
    pub fn new(max_capacity_bytes: u64) -> Self {
        let cache = CacheBuilder::new(max_capacity_bytes)
            .name("vortex-namespaced-segment-cache")
            .weigher(|_, buffer: &ByteBuffer| {
                u32::try_from(buffer.len().min(u32::MAX as usize)).vortex_expect("must fit")
            })
            .eviction_policy(EvictionPolicy::tiny_lfu())
            .build_with_hasher(FxBuildHasher);
        Self {
            inner: Arc::new(NamespacedMokaInner {
                cache,
                file_ids: DashMap::default(),
                next_file_id: AtomicU32::new(0),
            }),
        }
    }
}

impl SegmentCacheBuilder for NamespacedMokaSegmentCacheBuilder {
    fn cache_for(&self, file: &FileIdentity) -> SharedSegmentCache {
        let inner = &self.inner;
        let file_id = *inner
            .file_ids
            .entry(file.clone())
            .or_insert_with(|| inner.next_file_id.fetch_add(1, Ordering::Relaxed));
        Arc::new(NamespacedMokaSegmentCache {
            file_id,
            cache: inner.cache.clone(),
        })
    }
}

/// Per-file [`SegmentCache`] view returned by [`NamespacedMokaSegmentCacheBuilder`].
struct NamespacedMokaSegmentCache {
    file_id: u32,
    cache: Cache<(u32, u32), ByteBuffer, FxBuildHasher>,
}

#[async_trait]
impl SegmentCache for NamespacedMokaSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.cache.get(&(self.file_id, *id)).await)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.cache.insert((self.file_id, *id), buffer).await;
        Ok(())
    }
}
