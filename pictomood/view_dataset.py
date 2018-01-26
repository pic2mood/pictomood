"""
.. module:: view_dataset
    :synopsis: dataset viewer module

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Jan 20, 2018
"""

if __name__ == '__main__':

    import sys
    from pictomood import config
    import pictomood.utils

    if len(sys.argv) > 1:
        if sys.argv[1]:
            pictomood.utils.view_dataset(sys.argv[1])
