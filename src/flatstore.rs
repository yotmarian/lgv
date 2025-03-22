use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    marker::PhantomData,
    mem::{align_of, size_of},
    ops::Deref,
    path::Path,
};

use anyhow::{anyhow, Context, Result};
use log::info;
use zerocopy::{FromBytes, Immutable, IntoBytes};


pub struct Writer<T>(BufWriter<File>, PhantomData<fn(T)>);

impl<T: IntoBytes + Immutable> Writer<T> {
    pub fn open(path: &Path) -> Result<Self> {
        info!("Opening flatstore {path:?} for writing");

        Ok(Writer(
            BufWriter::new(
                OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(path)
                    .context("flatstore open writer")?,
            ),
            PhantomData,
        ))
    }

    pub fn write(&mut self, value: &T) -> Result<()> {
        self.0
            .write_all(value.as_bytes())
            .context("flatstore write")
    }
}


pub struct Reader<T>(File, PhantomData<fn() -> T>);

impl<T: IntoBytes + FromBytes> Reader<T> {
    pub fn open(path: &Path) -> Result<Self> {
        info!("Opening flatstore {path:?} for reading");

        let mut reader = Reader(
            OpenOptions::new()
                .read(true)
                .open(path)
                .context("flatstore open reader")?,
            PhantomData,
        );

        let size = reader.size()?;
        info!("Found {size} elements");
        Ok(reader)
    }

    pub fn size(&mut self) -> Result<u64> {
        let bytes = self.0.seek(SeekFrom::End(0)).context("flatstore size")?;
        Ok(bytes / size_of::<T>() as u64)
    }

    pub fn read(&mut self, idx: u64, buf: &mut T) -> Result<()> {
        self.0
            .seek(SeekFrom::Start(idx * size_of::<T>() as u64))
            .context("flatstore seek")?;

        self.0
            .read_exact(buf.as_mut_bytes())
            .context("flatstore read")
    }
}


pub struct Mmap<T>(memmap::Mmap, PhantomData<fn() -> T>);

impl<T> Mmap<T> {
    pub fn open(path: &Path) -> Result<Self> {
        info!("Opening flatstore {path:?} as a memory map");

        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .context("flatstore open reader")?;

        let mmap = Mmap(
            unsafe { memmap::Mmap::map(&file) }.context("flatstore map")?,
            PhantomData,
        );

        if mmap.0.len() % size_of::<T>() != 0 {
            Err(anyhow!(
                "flatstore: file size is not a multiple of record len"
            ))
        } else if !mmap.0.as_ptr().is_aligned_to(align_of::<T>()) {
            Err(anyhow!("flatstore: mmap is not properly aligned"))
        } else {
            Ok(mmap)
        }
    }
}

impl<T: FromBytes + Immutable> Deref for Mmap<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        <[T] as FromBytes>::ref_from_bytes(&self.0)
            .expect("flatstore: invariants broken since opened")
    }
}
