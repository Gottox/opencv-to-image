mod opencv;

use std::ops::Deref;

use bytemuck::{Pod, Zeroable};
use image::{ImageBuffer, Pixel};
use opencv::{boxed_ref::BoxedRef, core::DataType, core::Mat, prelude::*};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Opencv(#[from] opencv::Error),
    #[error("DimensionMissmatch: Expected array size: {exp}; actual: {0}", exp = *(.1) as i32 * *(.2) * *(.3))]
    DimensionMissmatch(usize, u8, i32, i32),
}
type Result<T> = std::result::Result<T, Error>;

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
struct PixelWrap<P: Copy + 'static>(P);

unsafe impl<P: Pixel> DataType for PixelWrap<P> {
    fn opencv_depth() -> i32 {
        0
    }
    fn opencv_channels() -> i32 {
        P::CHANNEL_COUNT as i32
    }
}
unsafe impl<P: Copy + 'static> Pod for PixelWrap<P> {}
unsafe impl<P: Copy> Zeroable for PixelWrap<P> {}

pub trait ToImage {
    fn as_image<P: Pixel<Subpixel = u8>>(
        &self,
    ) -> Result<ImageBuffer<P, &[P::Subpixel]>>;
    fn to_image<P: Pixel<Subpixel = u8>>(
        &self,
    ) -> Result<ImageBuffer<P, Vec<P::Subpixel>>>;
}

pub trait ToMat {
    fn as_mat(&self) -> Result<BoxedRef<Mat>>;
    fn to_mat(&self) -> Result<Mat> {
        Ok(self.as_mat()?.try_clone()?)
    }
}

impl<P, C> ToMat for ImageBuffer<P, C>
where
    P: Pixel<Subpixel = u8>,
    C: Deref<Target = [P::Subpixel]>,
    PixelWrap<P>: DataType + Pod + Zeroable,
{
    fn as_mat(&self) -> Result<BoxedRef<Mat>> {
        let data = self.as_raw().deref();
        let (width, height) = self.dimensions();
        let data: &[PixelWrap<P>] = bytemuck::cast_slice(data);

        Mat::new_rows_cols_with_data(height as i32, width as i32, data)
            .map_err(Error::Opencv)
    }
}

fn to_image<P: Pixel<Subpixel = u8>, C: Deref<Target = [P::Subpixel]>>(
    data: C,
    mat: &Mat,
) -> Result<ImageBuffer<P, C>> {
    let height = mat.rows();
    let width = mat.cols();
    let len = data.len();

    ImageBuffer::from_raw(width as u32, height as u32, data).ok_or(
        Error::DimensionMissmatch(len, P::CHANNEL_COUNT, width, height),
    )
}

impl ToImage for Mat {
    fn as_image<P: Pixel<Subpixel = u8>>(
        &self,
    ) -> Result<ImageBuffer<P, &[P::Subpixel]>> {
        to_image(self.data_bytes()?, self)
    }
    fn to_image<P: Pixel<Subpixel = u8>>(
        &self,
    ) -> Result<ImageBuffer<P, Vec<P::Subpixel>>> {
        to_image(self.data_bytes()?.to_vec(), self)
    }
}
