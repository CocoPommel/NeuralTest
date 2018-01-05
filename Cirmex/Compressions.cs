using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cirmex
{
    class Compressions
    {
        public static double rgbcomp(double x) //takes a rgb value and spits a unique value bewteen 0 and 1, so initial neuron activations don't suck (max value is 255 * 65536 + 255 * 256 + 255)
        {
            return x / 16777215;
        }

        public static double ReLU(double x) //determines how active neurons are for some input
        {
            if (x > 0)
            {
                return x;
            }
            return 0;
        }

        public static double derReLU(double x) //derivative of ReLU, defined 0 at 0 because it wouldnt exist otherwise and simplifies things
        {
            if (x > 0)
            {
                return 1;
            }
            return 0;
        }
    }
}
