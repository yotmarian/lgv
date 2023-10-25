use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write, Read, Seek, SeekFrom};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::Path;

use anyhow::{Result, Context};
use zerocopy::{AsBytes, FromBytes};


pub struct Writer<T>(BufWriter<File>, PhantomData<fn(T)>);

impl<T: AsBytes> Writer<T> {
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Writer(
            BufWriter::new(
                OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(path)?
            ),
            PhantomData,
        ))
    }

    pub fn write(&mut self, value: &T) -> Result<()> {
        self.0.write_all(value.as_bytes())
            .context("flatstore write")
    }
}


pub struct Reader<T>(File, PhantomData<fn() -> T>);

impl<T: AsBytes + FromBytes> Reader<T> {
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Reader(
            OpenOptions::new()
                .read(true)
                .open(path)?,
            PhantomData,
        ))
    }

    pub fn size(&mut self) -> Result<u64> {
        let bytes = self.0.seek(SeekFrom::End(0))
            .context("flatstore size")?;

        Ok(bytes / size_of::<T>() as u64)
    }

    pub fn read(&mut self, idx: u64, buf: &mut T) -> Result<()> {
        self.0.seek(SeekFrom::Start(idx * size_of::<T>() as u64))
            .context("flatstore seek")?;

        self.0.read_exact(buf.as_bytes_mut())
            .context("flatstore read")
    }
}
