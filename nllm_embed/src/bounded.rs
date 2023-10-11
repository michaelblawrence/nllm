pub struct BoundedValueLogger<T> {
    items: Vec<(usize, T)>,
    every_nth: usize,
    capacity: usize,
    index: usize,
    counter: usize,
}

impl<T: Ord> FromIterator<T> for BoundedValueLogger<T> {
    fn from_iter<A: IntoIterator<Item = T>>(iter: A) -> Self {
        let mut x = Self::new(0);
        for item in iter {
            x.insert(item);
        }
        let capacity = x.items.len();
        let next_capacity = (capacity as f64).log2().ceil().exp2() as usize;
        x.set_capacity(next_capacity);
        x
    }
}
impl<T> FromIterator<(usize, T)> for BoundedValueLogger<T> {
    fn from_iter<A: IntoIterator<Item = (usize, T)>>(iter: A) -> Self {
        let mut v: Vec<(usize, T)> = iter.into_iter().collect();
        v.sort_by_key(|(idx, _)| *idx);

        let mut x = Self::new(0);
        for (idx, item) in v.into_iter() {
            x.items.push((idx, item));
            x.counter = idx + 1;
        }

        let capacity = x.items.len();
        let next_capacity = (capacity as f64).log2().ceil().exp2() as usize;
        x.set_capacity(next_capacity);
        x
    }
}

impl<T> BoundedValueLogger<T> {
    const MIN_CAPACITY: usize = 16;

    pub fn new(capacity: usize) -> Self {
        Self {
            items: vec![],
            every_nth: 1,
            capacity: capacity.max(Self::MIN_CAPACITY),
            index: 0,
            counter: 0,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter().map(|(_, item)| item)
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity.max(Self::MIN_CAPACITY);
        let mut iter_left = 25;
        while iter_left > 0 {
            if !self.shrink_if_needed() {
                break;
            }
            iter_left -= 1;
        }
    }

    pub fn push(&mut self, item: T) -> bool {
        let added = match self.index {
            index if index == self.every_nth - 1 => {
                self.insert(item);
                self.shrink_if_needed();
                self.index = self.items.len() % self.every_nth;
                true
            }
            index if index < self.every_nth => {
                self.index += 1;
                false
            }
            _ => {
                self.index = 0;
                false
            }
        };

        added
    }

    fn insert(&mut self, item: T) {
        self.items.push((self.counter, item));
        self.counter += 1;
    }

    fn shrink_if_needed(&mut self) -> bool {
        if self.items.len() + self.every_nth <= self.capacity {
            return false;
        }

        self.every_nth += 1;

        let mut last_group_idx = None;
        let mut items = Vec::with_capacity(self.items.len());

        for (idx, item) in self.items.drain(..) {
            let group_idx = idx / self.every_nth;

            if last_group_idx.map_or(true, |last_idx| group_idx != last_idx) {
                items.push((idx, item));
            }

            last_group_idx = Some(group_idx);
        }

        let mean_stride = self.counter / self.capacity;
        self.every_nth = self.every_nth.max(mean_stride);

        self.items = items;

        true
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_logger_can_remove_excess_items() {
        use itertools::Itertools;

        let mut instance = BoundedValueLogger::new(6);
        instance.push(0);
        instance.push(1);
        instance.push(2);
        instance.push(3);
        instance.push(4);

        assert_eq!(instance.iter().cloned().collect_vec(), vec![0, 1, 2, 3, 4]);

        instance.push(5);
        assert_eq!(instance.iter().cloned().collect_vec(), vec![0, 2, 4]);
        
        instance.push(6);
        instance.push(7);
        assert_eq!(instance.iter().cloned().collect_vec(), vec![0, 2, 4, 6]);
        
        instance.push(8);
        assert_eq!(instance.iter().cloned().collect_vec(), vec![0, 4, 6]);
    }
}
