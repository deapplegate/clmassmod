'''Common utilities and things that are needed everywhere'''

############

import imp, sys

############

class NotConfiguredException(Exception): pass

####

def readConfiguration(configname):

    if 'currentconfig' in sys.modules:
        del sys.modules['currentconfig']

    configmodule = imp.load_source('currentconfig', configname)

    config = configmodule.__dict__

    runConfigure(config)

    return config

#######################

def runConfigure(config):

    toconfigure = []
    for val in config.itervalues():
        if hasattr(val, 'configure'):
            toconfigure.append(val)

    for val in toconfigure:
        val.configure(config)
    

########################


class Composite(object):

    def __init__(self, *pickers):
        self.pickers = pickers

    def configure(self, config):

        for picker in self.pickers:
            if hasattr(picker, 'configure'):
                picker.configure(config)

    def __call__(self, sim):

        curcat = sim
        for picker in self.pickers:
            curcat = picker(curcat)

        return curcat
