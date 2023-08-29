#[cfg(not(feature = "threadpool"))]
use std::cell::UnsafeCell;
#[cfg(feature = "threadpool")]
use std::sync::Mutex;

// TODO: evaluate perf for 'threadpool' feature
#[cfg(feature = "threadpool")]
#[derive(Debug, Default)]
pub(crate) struct MutexCell<T>(Mutex<T>);

#[cfg(feature = "threadpool")]
impl<'a, T> MutexCell<T> {
    pub fn new(value: T) -> Self {
        Self(Mutex::new(value))
    }

    pub fn with_inner<F: Fn(&mut T) -> O, O>(&'a self, func: F) -> O {
        let mut cell = self.0.lock().unwrap();
        func(&mut *cell)
    }
}

#[cfg(not(feature = "threadpool"))]
#[derive(Debug, Default)]
pub(crate) struct MutexCell<T>(UnsafeCell<T>);

#[cfg(not(feature = "threadpool"))]
impl<'a, T> MutexCell<T> {
    pub fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    pub fn with_inner<F: Fn(&mut T) -> O, O>(&'a self, func: F) -> O {
        let inner = unsafe { &mut *self.0.get() };
        func(inner)
    }
}

impl<T: Clone> Clone for MutexCell<T> {
    fn clone(&self) -> Self {
        Self::new(self.with_inner(|x| x.clone()))
    }
}