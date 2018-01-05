using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cirmex
{
    class ImageUtil
    {
        const System.Drawing.Imaging.ImageLockMode readOnly = System.Drawing.Imaging.ImageLockMode.ReadOnly; //shouldn't change this
        const System.Drawing.Imaging.PixelFormat Rgb24 = System.Drawing.Imaging.PixelFormat.Format24bppRgb; //change pixel data read in (8b per pixel for each color, w/o alpha)
        const int pixelWidth = 3;

        public static bool VerifyImage(String name)
        {
            try
            {
                using (Image n = Image.FromFile(name))
                { }
            }
            catch (OutOfMemoryException)
            {
                Console.WriteLine("Error: Invalid file format/excessively large image.");
                return false;
            }
            catch (System.IO.FileNotFoundException)
            {
                Console.WriteLine("File not found.");
                return false;
            }

            return true;
        }

        public static void getRGB(Bitmap image, int startX, int startY, int w, int h, int[] rgbArray, int offset, int scansize)
        {

            // Credit to Andy Hopper on StackOverflow
            if (image == null) throw new ArgumentNullException("image");
            if (rgbArray == null) throw new ArgumentNullException("rgbArray");
            Console.WriteLine(startX);
            Console.WriteLine(startY);
            Console.WriteLine(w);
            Console.WriteLine(h);
            if (startX < 0 || startX + w > image.Width) throw new ArgumentOutOfRangeException("startX");
            if (startY < 0 || startY + h > image.Height) throw new ArgumentOutOfRangeException("startY");
            if (w < 0 || w > scansize || w > image.Width) throw new ArgumentOutOfRangeException("w");
            if (h < 0 || (rgbArray.Length < offset + h * scansize) || h > image.Height) throw new ArgumentOutOfRangeException("h");

            var data = image.LockBits(new Rectangle(startX, startY, image.Width, image.Height), readOnly, Rgb24);
            try
            {
                byte[] pixelData = new Byte[data.Stride];
                for (int scanline = 0; scanline < data.Height; scanline++)
                {
                    Marshal.Copy(data.Scan0 + (scanline * data.Stride), pixelData, 0, data.Stride);
                    for (int pixeloffset = 0; pixeloffset < data.Width; pixeloffset++)
                    {
                        // PixelFormat.Format32bppRgb means the data is stored
                        // in memory as BGR. We want RGB, so we must do some 
                        // bit-shuffling.
                        rgbArray[offset + (scanline * scansize) + pixeloffset] =
                            (pixelData[pixeloffset * pixelWidth + 2] << 16) +   // R 
                            (pixelData[pixeloffset * pixelWidth + 1] << 8) +    // G
                            pixelData[pixeloffset * pixelWidth];                // B
                    }
                }
            }
            finally
            {
                image.UnlockBits(data);
            }
        }

        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }
    }
}