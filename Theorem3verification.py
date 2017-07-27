#===============================================================================
print "Welcome!\r\nThis program verifies Theorem 3."
print "[Tested with Python 2.7.6 and NumPy 1.8.0]"

#===============================================================================
from numpy import *

#===============================================================================
#
#   Itarators

def segments(word, j):
    """
    all ways to partition a word to t nonempty segments
    """
    if j == 1:
        yield [word]
    else:
        for i in range(1,len(word)):
            for x in segments(word[i:],j-1):
                yield [word[:i]] + x

def hypercube(n, zero = 0, one = 1):
    """
    all 0/1 vectors of size n
    """
    if n == 0:
        yield ()
    else:
        for s in hypercube(n-1, zero, one):
            yield s + (zero,)
            yield s + (one,)

def perms(items):
    """
    all permutations
    """
    if len(items) == 1:
        yield items
    for item in items[:]:
        items.remove(item)
        for perm in perms(items):
            yield [item] + perm
        items.append(item)

#===============================================================================
#
#   Diagrams

class Diagram:
    """
    A class for Gauss diagrams
    """

    def __init__(self, tips, signs = {}, coef = 1):
        """
        tips    list of integers, positive for arrow tail, negative for arrow head
        signs   dictionary, arrow number to sign
        coef    number
        """
        self.tips = tips
        self.signs = signs
        self.coef = coef
        self.arrows = [tip for tip in tips if tip > 0]

    def __repr__(self):
        return 'Diagram(%s, [%s], %s)' % (
            self.coef,
            ' '.join(map(str,self.tips)),
            ' '.join(['%d:%d'%(x,y) for x,y in self.signs.items()]))

    def multisigns(self):
        """
        produce list of diagrams multiplicative at signs
        """
        deg = len(self.arrows)
        return [Diagram(self.tips, dict(zip(self.arrows, signs)), prod(signs) * self.coef)
                for signs in hypercube(deg, 1, -1)]

#===============================================================================
#
#   Verification        

def order(this, other, same, parts = {(True,False):1,
                                      (False,True):2,
                                      (True,True):3,
                                      (False,False):4}):
    """
    By Proposition 10, monotone in the order that this segment meets others.
    
    this, other     segment indexes
    same            whether parity of segments agrees
    """
    return parts[this > other, same], -other

def calc_min_degree(sigma, formula):
    """
    Used to check Lemma 16, for given sigma and formula.

    simga       permutation
    formula     list of diagrams
    """

    # initialize coefficients vector    
    c = zeros((2,)*len(sigma))

    # sum over diagrams
    for D in formula:

        # and over all ways to divide diagram to segments
        for S in segments(D.tips, len(sigma)):

            # prepare arrow tip finder: number -> segment
            w = {a:i for i,s in enumerate(S) for a in s}

            # check that all arrows point upper to lower
            if all([sigma[w[a]] > sigma[w[-a]] for a in D.arrows]):

                # all 0/1 parity vectors
                for E in hypercube(len(sigma)):

                    # check that parities are compatible with the arrow signs
                    if all([(-1)**(E[w[a]]+E[w[-a]]) * sign(w[-a]-w[a])
                            == D.signs[a] for a in D.arrows]):

                        # check order of arrow tips in same segment
                        if all([order(w[t1],w[-t1],E[w[t1]]==E[w[-t1]])
                                < order(w[t2],w[-t2],E[w[t2]]==E[w[-t2]])
                                for s in S for t1,t2 in zip(s,s[1:])]):

                            # we have compatibility
                            c[E] += D.coef

    # discrete Fourier transform over (Z_2)^j
    Fc = fft.fftn(c).real

    # find minimum degree
    return min([sum(E) for E in hypercube(len(sigma)) if Fc[E]] + [inf])

def verify(formula):
    """
    Verify all cases of Lemma 16.
    """
    deg = max([len(D.arrows) for D in formula])
    for j in range(2*deg,deg,-1):
        for sigma in perms(range(j)):
            if calc_min_degree(sigma, formula) < 2*(j-deg):
                return False
    return True

#===============================================================================
#
#   Formulas

c2_down = Diagram([1,-2,-1,2]).multisigns()
c2_up = Diagram([-1,2,1,-2]).multisigns()

# twice v3
v3_GPV = Diagram([1, -2, -3, -1, 2, 3]).multisigns() + \
         Diagram([-1, 2, 3, 1, -2, -3]).multisigns() + \
         Diagram([1, -2, 3, -1, 2, -3]).multisigns() + \
         Diagram([-1, 2, -3, 1, -2, 3]).multisigns() + \
         Diagram([1, -2, -3, -1, 3, 2]).multisigns() + \
         Diagram([-1, 2, 3, 1, -3, -2]).multisigns() + \
         Diagram([1, -2, -3, 2, -1, 3]).multisigns() + \
         Diagram([-1, 2, 3, -2, 1, -3]).multisigns() + \
         Diagram([1, -2, -1, 3, 2, -3]).multisigns() + \
         Diagram([-1, 2, 1, -3, -2, 3]).multisigns() + [
             Diagram([-1, 2, 1, -2], {1:+1, 2:+1}, 1),
             Diagram([-1, 2, 1, -2], {1:-1, 2:+1}, -1),
             Diagram([1, -2, -1, 2], {1:+1, 2:-1}, 1),
             Diagram([1, -2, -1, 2], {1:-1, 2:-1}, -1)]

# twice v3
v3_PV = Diagram([-1,-2,1,-3,2,3]).multisigns() + \
        Diagram([-2,1,-3,2,3,-1]).multisigns() + \
        Diagram([1,-3,2,3,-1,-2]).multisigns() + \
        Diagram([-3,2,3,-1,-2,1]).multisigns() + \
        Diagram([2,3,-1,-2,1,-3]).multisigns() + \
        Diagram([3,-1,-2,1,-3,2]).multisigns() + \
        Diagram([1,-2,3,-1,2,-3], coef = 2).multisigns() + \
        Diagram([-1,2,-3,1,-2,3], coef = 2).multisigns()

# twice v3
v3_W = Diagram([-1,-2,3,1,-3,2],coef=1).multisigns() + \
       Diagram([1,2,-3,-1,3,-2],coef=1).multisigns() + \
       Diagram([-1,2,3,-2,1,-3],coef=1).multisigns() + \
       Diagram([1,-2,-3,2,-1,3],coef=1).multisigns() + \
       Diagram([-1,-2,1,3,2,-3],coef=1).multisigns() + \
       Diagram([-1,2,1,3,-2,-3],coef=-1).multisigns() + \
       Diagram([-1,2,1,-3,-2,3],coef=1).multisigns() + \
       Diagram([1,-2,-1,3,2,-3],coef=-1).multisigns() + \
       Diagram([1,2,-1,3,-2,-3],coef=2).multisigns() + \
       Diagram([1,-2,3,-1,2,-3], coef = 2).multisigns() + \
       Diagram([-1,2,-3,1,-2,3], coef = 2).multisigns()

# just v3
v3_CP = Diagram([-1,2,3,1,-2,-3]).multisigns() + \
        Diagram([-1,-2,3,1,2,-3]).multisigns() + \
        Diagram([-1,-2,3,2,1,-3]).multisigns() + \
        Diagram([-1,-2,1,3,2,-3]).multisigns() + \
        Diagram([-1,2,3,1,-3,-2]).multisigns() + [
             Diagram([-1,2,1,-2],{1:+1,2:+1},1),
             Diagram([-1,2,1,-2],{1:-1,2:-1},-1)]

writhe = Diagram([1,-1],coef=1).multisigns() + \
         Diagram([-1,1],coef=1).multisigns()

#===============================================================================
#
#   Main


if __name__ == '__main__':
    print "Theorem 3 is:",
    print verify(v3_GPV)
    temp = raw_input()

                
#===============================================================================
#
#   Knots

unknot = Diagram([],{})
positive_trefoil = Diagram([1,-2,3,-1,2,-3],{1:1,2:1,3:1})
negative_trefoil = Diagram([1,-2,3,-1,2,-3],{1:-1,2:-1,3:-1})
figure_eight = Diagram([1,-2,3,-4,2,-1,4,-3],{1:-1,2:-1,3:1,4:1})

def matchs(small, big,
           pb = 0, ps = 0, res = []):
    """
    find all sub arrow diagrams with matching tips
    """
    if ps == len(small):
        yield res
    else:
        ps1 = small.index(-small[ps])
        if ps1 < ps:
            i = big.index(-res[ps1])
            if i >= pb:
                for m in matchs(small, big, i+1, ps+1, res+[big[i]]):
                    yield m
        else:
            for i in range(pb, len(big)):
                if big[i] * small[ps] > 0:
                    for m in matchs(small, big, i+1, ps+1, res+[big[i]]):
                        yield m                       

def pairing(formula, knot):
    """
    compute: < formula , knot >
    """
    if not isinstance(formula, list):
        formula = [formula]
    value = 0
    for diagram in formula:
        for match in matchs(diagram.tips, knot.tips):
            if all([diagram.signs[t1] == knot.signs[t2]
                    for t1,t2 in zip(diagram.tips, match) if t1 > 0]):
                value += diagram.coef * knot.coef
    return value             

def petaluma(perm):
    """
    permutation to gauss diagram
    """
    n = len(perm)
    tips = [0]*(n*(n-3))
    signs = {}
    for i in range(n):
        for j in range(2,n-1):
            p = i*(n-3) + (j%2)*(n-3)/2 + (n-3)/2 - 1 - (j-2)/2
            a,b = sorted([i,(i+j)%n])
            k = a*n+b
            tips[p] = k * (-1) ** (perm[i] < perm[(i+j)%n])
            signs[k] = (-1) ** ((perm[i] > perm[(i+j)%n]) + j)
    return Diagram(tips, signs)

