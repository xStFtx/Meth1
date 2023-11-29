from sympy import symbols, Basic

# Define the coefficients β1, β2, ..., β19 as symbols
coefficients = symbols('β1:20')

# Define a class for our custom basis vectors
class BasisVector(Basic):
    def __init__(self, name):
        self.name = name

    def __mul__(self, other):
        if isinstance(other, BasisVector):
            return self.multiply_basis_vectors(self, other)
        else:
            raise ValueError("Can only multiply with another BasisVector object.")

    @staticmethod
    def multiply_basis_vectors(basis1, basis2):
        # Implement the rules as specified in the image
        rules = {
            ('e_a', 'e_c'): coefficients[2] * BasisVector('e_b'),
            ('e_a', 'e_b'): coefficients[1] * BasisVector('e_c'),
            # ... other rules ...
        }
        return rules.get((basis1.name, basis2.name), None)

    def __repr__(self):
        return self.name

# Instantiate our basis vectors
e_a = BasisVector('e_a')
e_b = BasisVector('e_b')
e_c = BasisVector('e_c')
em1 = BasisVector('em1')
em2 = BasisVector('em2')

# Now let's try to multiply e_a with e_c
product = e_a * e_c
print(product)
