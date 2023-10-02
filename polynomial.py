import numpy as np
import sympy

class Polynomial():
    def __init__(self, array):
        self.coff = array
        self.coefficients = self.coff
        self.order = len(array)
        
    @staticmethod
    def from_string(string):
        if 'x' not in string:
            array = np.array([int(string)])
        else:
            my_polyog = string
            my_poly = sympy.polys.polytools.poly_from_expr(my_polyog)
            array = np.array(my_poly[0].all_coeffs())

        return Polynomial(array)
    
    def __repr__(self):
        string = str(self.coff) 
        return string
    
    
    def __eq__(self, other):
        
        
        diff = len(self.coff)-len(other.coff)
        if len(self.coff)>len(other.coff):
            other.coff = np.pad(other.coff, ((abs(diff)),0))
        elif len(self.coff)<len(other.coff):
            self.coff = np.pad(self.coff, (abs(diff),0))
        
        return all(self.coff == other.coff)

    def __add__(self, other):
        
        diff = len(self.coff)-len(other.coff)
        if len(self.coff)>len(other.coff):
            other.coff = np.pad(other.coff, ((abs(diff)),0))
        elif len(self.coff)<len(other.coff):
            self.coff = np.pad(self.coff, (abs(diff),0))
        
            
        addition = self.coff + other.coff
        return Polynomial(addition)
    
    def __sub__(self, other):
        
        diff = len(self.coff)-len(other.coff)
        if len(self.coff)>len(other.coff):
            other.coff = np.pad(other.coff, ((abs(diff)),0))
        elif len(self.coff)<len(other.coff):
            self.coff = np.pad(self.coff, (abs(diff),0))
        
            
        sub = self.coff - other.coff
        
        return Polynomial(sub)
    
    def __mul__(self, other):
        
        x = sympy.Symbol('x')
        s1 = sympy.Poly.from_list(list(self.coff), x)
        o1 = sympy.Poly.from_list(list(other.coff), x)
        c1 = s1*o1
        array = np.array(c1.all_coeffs())

       
        return Polynomial(array)
    
    
    def __truediv__(self,other):
        return RationalPolynomial(self,other)

class RationalPolynomial():

    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
        self._reduce()
        
    def __eq__(self, other):
        if self.numerator == other.numerator:
            if self.denominator == other.denominator:
                return True
        return 
    def __repr__(self):
        string = str(self.numerator) + " / " + str(self.denominator)
        return string



    @staticmethod
    def from_string(string):
        numerator, denominator = string.split("/")
        return RationalPolynomial(Polynomial.from_string(numerator.strip('()')), Polynomial.from_string(denominator.strip('()')))
    
    def _reduce(self):
      
        x = sympy.symbols('x')
        s1 = sympy.Poly.from_list(list(self.numerator.coff),x)
        s2 = sympy.Poly.from_list(list(self.denominator.coff),x)
        s3 = s1/s2
        s3 = sympy.cancel(s3)
        s3 = str(s3)
        s3 = s3.split(("/"))
        if len(s3)==1:
            self.numerator = Polynomial.from_string(s3[0])
            self.denominator = Polynomial.from_string("1")
        else:
            self.numerator = Polynomial.from_string(s3[0])
            self.denominator = Polynomial.from_string(s3[1])
            
        
    
    def __add__(self, other):
        x = sympy.symbols('x')
        s1 = sympy.Poly.from_list(list(self.numerator.coff),x)
        s2 = sympy.Poly.from_list(list(self.denominator.coff),x)
        s3 = sympy.Poly.from_list(list(other.numerator.coff),x)
        s4 = sympy.Poly.from_list(list(other.denominator.coff),x)
        pol1 = s1/s2
        pol2 = s3/s4
        pol = pol1+pol2
        pol = sympy.simplify(pol)
        pol = str(pol)
        s3 = pol.split(("/"))
        if len(s3)==1:
            numerator = s3[0]
            denominator = "1"
        else:
            numerator = s3[0]
            denominator = s3[1]
          
        return RationalPolynomial(Polynomial.from_string(numerator), Polynomial.from_string(denominator))
    
    def __sub__(self, other):
        x = sympy.symbols('x')
        s1 = sympy.Poly.from_list(list(self.numerator.coff),x)
        s2 = sympy.Poly.from_list(list(self.denominator.coff),x)
        s3 = sympy.Poly.from_list(list(other.numerator.coff),x)
        s4 = sympy.Poly.from_list(list(other.denominator.coff),x)
        pol1 = s1/s2
        pol2 = s3/s4
        pol = (s1*s4-s3*s2)/(s2*s4)
        pol = str(pol)
        s3 = pol.split(("/"))
        if len(s3)==1:
            numerator = s3[0]
            denominator = "1"
        else:
            numerator = s3[0]
            denominator = s3[1]
          
        return RationalPolynomial(Polynomial.from_string(numerator), Polynomial.from_string(denominator))
    
    def __mul__(self, other):
        x = sympy.symbols('x')
        s1 = sympy.Poly.from_list(list(self.numerator.coff),x)
        s2 = sympy.Poly.from_list(list(self.denominator.coff),x)
        s3 = sympy.Poly.from_list(list(other.numerator.coff),x)
        s4 = sympy.Poly.from_list(list(other.denominator.coff),x)
        pol1 = s1/s2
        pol2 = s3/s4
        pol = pol1*pol2
        pol = sympy.simplify(pol)
        pol = str(pol)
        s3 = pol.split(("/"))
        if len(s3)==1:
            numerator = s3[0]
            denominator = "1"
        else:
            numerator = s3[0]
            denominator = s3[1]
          
        return RationalPolynomial(Polynomial.from_string(numerator), Polynomial.from_string(denominator))
    def __truediv__(self, other):
        x = sympy.symbols('x')
        s1 = sympy.Poly.from_list(list(self.numerator.coff),x)
        s2 = sympy.Poly.from_list(list(self.denominator.coff),x)
        s3 = sympy.Poly.from_list(list(other.numerator.coff),x)
        s4 = sympy.Poly.from_list(list(other.denominator.coff),x)
        pol1 = s1/s2
        pol2 = s3/s4
        pol = pol1/pol2
        pol = sympy.simplify(pol)
        pol = str(pol)
        s3 = pol.split(("/"))
        if len(s3)==1:
            numerator = s3[0]
            denominator = "1"
        else:
            numerator = s3[0]
            denominator = s3[1]
          
        return RationalPolynomial(Polynomial.from_string(numerator), Polynomial.from_string(denominator))

  

