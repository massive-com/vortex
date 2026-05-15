// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::dtype::DType;
use crate::stats::ArrayStats;
use crate::{ArrayRef, DynArray};

#[derive(Clone, Debug)]
pub struct ReversedArray {
    pub(super) child: ArrayRef,
    pub(super) dtype: DType,
    pub(super) len: usize,
    pub(super) stats: ArrayStats,
}

impl ReversedArray {
    /// Wraps `child` in a [`ReversedArray`].
    pub fn try_new(child: ArrayRef) -> VortexResult<Self> {
        let dtype = child.dtype().clone();
        let len = child.len();
        Ok(Self {
            child,
            dtype,
            len,
            stats: ArrayStats::default(),
        })
    }

    /// Wraps `child` in a [`ReversedArray`].
    pub fn new(child: ArrayRef) -> Self {
        Self::try_new(child).vortex_expect("failed to construct ReversedArray")
    }

    /// Wraps `child` in a [`ReversedArray`] without validation.
    ///
    /// # Safety
    ///
    /// Caller must ensure `child` is a valid array.1
    pub unsafe fn new_unchecked(child: ArrayRef) -> Self {
        #[cfg(debug_assertions)]
        Self::validate(&child, child.dtype(), child.len())
            .vortex_expect("[Debug Assertion]: Invalid `ReversedArray` parameter");

        let dtype = child.dtype().clone();
        let len = child.len();
        Self {
            child,
            dtype,
            len,
            stats: ArrayStats::default(),
        }
    }

    /// Returns the inner array whose elements will be yielded in reverse order.
    pub fn child(&self) -> &ArrayRef {
        &self.child
    }

    /// Validates the components that would be used to construct a [`ReversedArray`].
    pub fn validate(child: &ArrayRef, dtype: &DType, len: usize) -> VortexResult<()> {
        vortex_ensure!(
            child.dtype() == dtype,
            "ReversedArray dtype {} does not match child dtype {}",
            dtype,
            child.dtype(),
        );
        vortex_ensure!(
            child.len() == len,
            "ReversedArray length {} does not match child length {}",
            len,
            child.len(),
        );
        Ok(())
    }
}
