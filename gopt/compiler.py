import numba
import numpy as np
import logging


# This is the compiler class of GOPT
# Its main use is to let the user configure some potentially preformance
# impacting settings
class Compiler:

    # Will not trigger any compilation
    # It makes the code slow but might give better error messages
    debug = False

    # Whether to inline functions (see numba documentation)
    inline = 'always'
    #
    # Whether to use llvm fastmath option (see numba documentation)
    fastmath = True

    @classmethod
    def getLogger(cls, clz):
        logger = logging.getLogger('gopt.compiler')
        logger = logger.getChild(clz)
        return logger

    @classmethod
    def ufunc(cls, code):
        return cls.jit('ufunc', f'{code.__name__}', None, code)

    @classmethod
    def generate_allocator(cls, clz, ntype):
        cls.getLogger(clz).info('Generating allocator')

        if ntype is None:
            def allocator(_):
                return None
        else:
            def allocator(size=1):
                return np.empty(shape=size, dtype=ntype)

        return allocator

    @classmethod
    def jit(cls, clz, name, signature, code):
        if cls.debug:
            return code

        cls.getLogger(clz).info(f'Compiling {name}')
        return numba.njit(signature, inline=cls.inline,
                          fastmath=cls.fastmath)(code)


# This is a class extended by anything that can be compiled
# It's just to avoid repeating myself
class Compilable:

    ######
    # Should be implemented by deriving classes
    ######

    @classmethod
    def compile(cls):
        raise NotImplementedError

    ######
    # GOPT internals, do not overwrite!
    ######

    # Will contain a dict of compiled functions
    _compiled = None

    @classmethod
    def is_compiled(cls):
        return cls._compiled is not None
