import textwrap
import strax
import straxbra

export, __all__ = strax.exporter()

# pretty much just taken from straxen
ma_doc = """
This is a straxbra tiny-analysis.
The method takes run_id as its only positionsal argument,
and additional arguments through keywords only.

The function requires the data types: {requires}.
This data will be loaded automatically.

"""

@export
def tiny_analysis(requires=tuple()):
    def decorator(f):
        def wrapped_f(context, run_id, **kwargs):

            for dkind in ['peak', 'event']:
                kinds = [req for req in requires if dkind in req]
                if dkind == 'peak':
                    peaks = None if not kinds else context.get_array(run_id, kinds)
                else:
                    events = None if not kinds else context.get_df(run_id, kinds)

            if peaks is not None:
                kwargs['peaks'] = peaks
            if events is not None:
                kwargs['events'] = events

            return f(**kwargs)

        wrapped_f.__name__ = f.__name__

        if hasattr(f, '__doc__') and f.__doc__:
            doc_lines = f.__doc__.splitlines()
            wrapped_f.__doc__ = (doc_lines[0]
                                 + '\n'
                                 +textwrap.dedent(
                                  '\n'.join(doc_lines[1:])))
        else:
            wrapped_f.__doc__ = \
                'There is no proper docstring for this tiny-analysis ¯\_(ツ)_/¯'

        # wrapped_f.__doc__ += ma_doc.format(requires=', '.join(requires))

        strax.Context.add_method(wrapped_f)
        return wrapped_f
    return decorator
