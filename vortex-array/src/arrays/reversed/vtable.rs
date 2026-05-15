// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::Precision;
use crate::arrays::reversed::array::ReversedArray;
use crate::arrays::reversed::execute::reverse_canonical;
use crate::arrays::reversed::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;

vtable!(Reversed);

/// Encoding tag for [`ReversedArray`].
#[derive(Clone, Debug)]
pub struct Reversed;

impl Reversed {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.reversed");
}

impl VTable for Reversed {
    type Array = ReversedArray;
    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(_array: &Self::Array) -> &Self {
        &Reversed
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ReversedArray) -> usize {
        array.len
    }

    fn dtype(array: &ReversedArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ReversedArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &ReversedArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.len.hash(state);
        array.child.array_hash(state, precision);
    }

    fn array_eq(array: &ReversedArray, other: &ReversedArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.len == other.len
            && array.child.array_eq(&other.child, precision)
    }

    fn nbuffers(_array: &ReversedArray) -> usize {
        0
    }

    fn buffer(_array: &ReversedArray, idx: usize) -> BufferHandle {
        vortex_panic!("ReversedArray has no buffers (index {idx})")
    }

    fn buffer_name(_array: &ReversedArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &ReversedArray) -> usize {
        1
    }

    fn child(array: &ReversedArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.child.clone(),
            _ => vortex_panic!("ReversedArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &ReversedArray, idx: usize) -> String {
        match idx {
            0 => "child".to_string(),
            _ => vortex_panic!("ReversedArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(_array: &ReversedArray) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("ReversedArray is not serializable")
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        vortex_bail!("ReversedArray is not serializable")
    }

    fn build(
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ReversedArray> {
        vortex_ensure!(
            children.len() == 1,
            "ReversedArray expects exactly 1 child, got {}",
            children.len()
        );
        let child = children.get(0, dtype, len)?;
        ReversedArray::try_new(child)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 1,
            "ReversedArray expects exactly 1 child, got {}",
            children.len()
        );
        let child = children
            .into_iter()
            .next()
            .vortex_expect("children length already validated");
        ReversedArray::validate(&child, &array.dtype, array.len)?;
        array.child = child;
        Ok(())
    }

    fn execute(array: Arc<Self::Array>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        reverse_canonical(&array.child, ctx).map(ExecutionResult::done)
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<Reversed> for Reversed {
    fn scalar_at(array: &ReversedArray, index: usize) -> VortexResult<Scalar> {
        let reversed_index = array.len - 1 - index;
        array.child.scalar_at(reversed_index)
    }
}

impl ValidityVTable<Reversed> for Reversed {
    fn validity(array: &ReversedArray) -> VortexResult<Validity> {
        let inner = array.child.validity()?;
        match inner {
            Validity::NonNullable => Ok(Validity::NonNullable),
            Validity::AllValid => Ok(Validity::AllValid),
            Validity::AllInvalid => Ok(Validity::AllInvalid),
            Validity::Array(arr) => Ok(Validity::Array(arr.reverse()?)),
        }
    }
}
