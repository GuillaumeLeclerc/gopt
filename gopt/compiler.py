import numba
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
    def ufunc(cls, code):
        return cls.jit('ufunc', f'{code.__name__}', None, code)

    @classmethod
    def jit(cls, clz, name, signature, code):
        if cls.debug:
            return code

        logger = logging.getLogger('gopt.compiler')
        logger = logger.getChild(clz)
        logger.info(f'Compiling {name}')
        return numba.njit(signature, inline=cls.inline,
                          fastmath=cls.fastmath)(code)
