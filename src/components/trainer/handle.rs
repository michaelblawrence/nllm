
use std::{self, ops::Deref, rc::Rc};

use wasm_bindgen::__rt::WasmRefCell;

use crate::ml::{embeddings::Embedding, JsRng};

#[derive(Clone)]
pub struct EmbeddingHandle(Rc<WasmRefCell<Embedding>>, usize);

impl Default for EmbeddingHandle {
    fn default() -> Self {
        let inner = Self::default_inner();
        Self::new(inner)
    }
}

impl EmbeddingHandle {
    pub fn new(inner: Embedding) -> Self {
        Self(Rc::new(WasmRefCell::new(inner)), 0)
    }
    pub fn replace(&self, inner: Embedding) -> Self {
        *self.0.borrow_mut() = inner;
        Self(self.0.clone(), self.1 + 1)
    }
    pub fn replace_with<F: FnOnce(Embedding) -> Embedding>(&self, replace_fn: F) -> Self {
        // panic safety?
        let inner = std::mem::replace(&mut *self.0.borrow_mut(), Self::default_inner());
        let inner = replace_fn(inner);
        let _ = std::mem::replace(&mut *self.0.borrow_mut(), inner);
        Self(self.0.clone(), self.1 + 1)
    }
    pub fn tick(&self) -> Self {
        Self(self.0.clone(), self.1 + 1)
    }
    fn default_inner() -> Embedding {
        Embedding::new(
            Default::default(),
            Default::default(),
            1,
            vec![],
            Rc::new(JsRng::default()),
        )
    }
}

impl PartialEq for EmbeddingHandle {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl Deref for EmbeddingHandle {
    type Target = WasmRefCell<Embedding>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}


#[derive(Clone)]
pub struct RefCellHandle<T>(Rc<WasmRefCell<T>>, usize);

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