using System;
using MathNet.Numerics.LinearAlgebra;
using System.Drawing;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cirmex
{
    class Program
    {
        static void Main(string[] args)
        {
            String name;
            Bitmap img;
            int resizedWidth = 1000;
            int resizedHeight = 1000;
            double[] expected = { 0, 0, 0, 0, 0, 0, 1 };

            do
            {
                Console.WriteLine("Enter an image:"); //get image filepath
                name = Console.ReadLine();
            } while (!ImageUtil.VerifyImage(name)); //make sure image is actually image and not large
            
            img = Bitmap.FromFile(name) as Bitmap; //do this again here so we have it but know it's legit
            Bitmap resizedImg = ImageUtil.ResizeImage(img, resizedWidth, resizedHeight); //resize the image so it doesn't take multiple millenia
            resizedImg.Save("resized.png", System.Drawing.Imaging.ImageFormat.Png);
            int[] rgb = new int[resizedWidth * resizedHeight]; //holds int rgb values
            ImageUtil.getRGB(resizedImg, 0, 0, resizedWidth, resizedHeight, rgb, 0, resizedWidth);

            double[] input = new double[resizedWidth * resizedHeight]; //since we need doubles for compression to not have floating point errors
            for (int i = 0; i < rgb.Length; i++)
            {
                input[i] = Compressions.rgbcomp(rgb[i]); //compress int rgb values and put them into input
            }

            Layer initial = new Layer(input); //create initial layer with compressed activation values
            Console.WriteLine("How many hidden layers?");
            int numhidden = Convert.ToInt16(Console.ReadLine());
            int[] hidden = new int[numhidden];
            for(int i = 0; i < numhidden; i++)
            {
                Console.WriteLine("How many neurons in hidden layer " + i + "?");
                hidden[i] = Convert.ToInt16(Console.ReadLine());
            }

            Network neural = new Network(resizedWidth * resizedHeight, hidden, 7);
            neural.randomizeAll();

            for (int i = 0; i < 100; i++)
            {
                int errorCode = runNetwork(neural, input, expected);
                if(errorCode == -1)
                {
                    Console.WriteLine("Intended output is a dead neuron, killing network.");
                    Console.ReadKey();
                    return;
                }
                else if(errorCode == -2)
                {
                    Console.WriteLine("Every output is a dead neuron, killing network.");
                    Console.ReadKey();
                    return;
                }
            }

            Console.ReadKey();
        }

        static int runNetwork(Network neural, double[] input, double[] expected)
        {
            neural.eval(Vector<double>.Build.Dense(input));
            int ret = neural.applySteps(neural.backpropogate(expected), expected);
            Console.WriteLine(neural.costFun(expected));
            return ret;
        }
    }
}