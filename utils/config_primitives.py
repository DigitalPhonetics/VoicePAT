class IncompleteConfigException(Exception):
    pass


class Todo:
    def __init__(self, *args, **kwargs):
        raise IncompleteConfigException(
            'You must complete the config files before execution. ' 
            'Find lines with "TODO" and enter the appropriate information.'
        )
