mod opencv;

use bytemuck::{Pod, Zeroable};
use image::{ImageBuffer, Pixel, Rgb, Rgba};
use opencv::{boxed_ref::BoxedRef, core::DataType, core::Mat, prelude::*};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Opencv(#[from] opencv::Error),
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
struct PixelBuf<const N: usize>([u8; N]);

unsafe impl<const N: usize> DataType for PixelBuf<N> {
    fn opencv_depth() -> i32 {
        0
    }

    fn opencv_channels() -> i32 {
        N as i32
    }
}
unsafe impl<const N: usize> Pod for PixelBuf<N> {}
unsafe impl<const N: usize> Zeroable for PixelBuf<N> {}

trait HasPixelBuf {
    type PixelBuf;
}

macro_rules! impl_pixel_buf {
    ($format: ty, $channels: expr) => {
        impl HasPixelBuf for $format {
            type PixelBuf = PixelBuf<$channels>;
        }
        const _: () = {
            type Format = $format;
            static_assertions::const_assert!(Format::CHANNEL_COUNT == $channels);
        };
    };
}

impl_pixel_buf!(Rgb<u8>, 3);
impl_pixel_buf!(Rgba<u8>, 4);

pub trait ToMat {
    fn as_mat(&self) -> Result<BoxedRef<Mat>, Error>;
    fn to_mat(&self) -> Result<Mat, Error> {
        Ok(self.as_mat()?.try_clone()?)
    }
}

impl<P, C> ToMat for ImageBuffer<P, C>
where
    P: Pixel<Subpixel = u8> + HasPixelBuf,
    C: std::ops::Deref<Target = [P::Subpixel]>,
    P::PixelBuf: DataType + Pod + Zeroable,
{
    fn as_mat(&self) -> Result<BoxedRef<Mat>, Error> {
        let data = self.as_raw().deref();
        let (width, height) = self.dimensions();
        let data: &[P::PixelBuf] = bytemuck::cast_slice(data);

        Ok(Mat::new_rows_cols_with_data(
            height as i32,
            width as i32,
            data,
        )?)
    }
}
