use std::{self, ops::Deref, rc::Rc};

use wasm_bindgen::__rt::WasmRefCell;

use crate::ml::embeddings::Embedding;

pub type EmbeddingHandle = RefCellHandle<Embedding>;

pub struct RefCellHandle<T>(Rc<WasmRefCell<T>>, usize);

impl<T> Clone for RefCellHandle<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone())
    }
}

impl<T: Default> Default for RefCellHandle<T> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<T> RefCellHandle<T> {
    pub fn new(inner: T) -> Self {
        Self(Rc::new(WasmRefCell::new(inner)), 0)
    }
    pub fn replace(&self, inner: T) -> Self {
        *self.0.borrow_mut() = inner;
        Self(self.0.clone(), self.1 + 1)
    }
    pub fn tick(&self) -> Self {
        Self(self.0.clone(), self.1 + 1)
    }
}

impl<T: Default> RefCellHandle<T> {
    pub fn replace_with<F: FnOnce(T) -> T>(&self, replace_fn: F) -> Self {
        // panic safety?
        let inner = std::mem::take(&mut *self.0.borrow_mut());
        let inner = replace_fn(inner);
        let _ = std::mem::replace(&mut *self.0.borrow_mut(), inner);
        Self(self.0.clone(), self.1 + 1)
    }
}

impl<T> PartialEq for RefCellHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<T> Deref for RefCellHandle<T> {
    type Target = WasmRefCell<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
