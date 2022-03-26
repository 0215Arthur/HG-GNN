"""
Config.py : given the utility tool of setting configuration for models.

We follow the design and codes of NeuRec
@author: Zhongchuan Sun https://github.com/wubinzzu/NeuRec

Available Object:
    Configurator
"""
import os
import sys
from configparser import ConfigParser
from collections import OrderedDict

class Configurator(object):
    """The configuration class for rec models.

    This class read arguments from configuration file and parse the arguments via the configparser tool.

    In our project, we only design the from configuration file, 
    not support parse arguments form command line.

    There are two types configuration file: 
        1: general configuration file, 
        which setting the basic and universal arguments for model training and data processing
        2. specific configuration file,
        which setting the hyperparameters for specific model 


    Methods:


    Usage:

    
    """

    def __init__(self,config_file,default_section="default"):
        """Iniialization

        Params:
            config_file (str): general  configuration file;
            default_section (str): The default section for configparser

        """

        if not os.path.isfile(config_file):
            raise FileNotFoundError("There is not config file named '%s'!" % config_file)

        self._default_section = default_section
        self.cmd_arg = self._read_cmd_arg()
        self.lib_arg = self._read_config_file(config_file)
        config_dir = self.lib_arg["config_dir"]
        # load the specific config_file for recommender model
        if 'recommender' in self.cmd_arg:
            model_name=self.cmd_arg["recommender"]
        else:
            model_name = self.lib_arg["recommender"]
        arg_file = os.path.join(config_dir, model_name+'.ini')
        self.alg_arg = self._read_config_file(arg_file)
        
    
    def _read_config_file(self,config_file):

        """read & parse the config_file

        Params:
           config_file(str): the config filename
        
        return:
           config_args(dict): config arguments dict
        
        """

        config=ConfigParser()
        # load the config and check the section
        config.read(config_file)
        sections=config.sections()
        if len(sections)<1:
            raise ValueError("%s is empty file!"%config_file)
        elif len(sections)==1:
            config_sec=sections[0]
        elif self._default_section in sections:
            config_sec =self._default_section
        else:
            raise ValueError("%s not find the default section %s"%config_file,self._default_section)
        config_args=OrderedDict(config[config_sec].items())
        
        return config_args
    
    def change_attr(self,item,value):
        if not isinstance(item,str):
            raise TypeError("index must be a str!")
        if item in self.lib_arg:
            self.lib_arg[item]=str(value)
        elif item in self.alg_arg:
            self.alg_arg[item]=str(value)
        else:
            raise KeyError('Not find the parameter %s'%item)
    
    def _read_cmd_arg(self):
        cmd_arg = OrderedDict()
        
        if "ipykernel_launcher" not in sys.argv[0]:
            if len(sys.argv[1:])%2:
                raise SyntaxError("Terrible Commend arg format")
            arg_name=None
            for i,arg in enumerate(sys.argv[1:]):
                if i%2==0:         
                    if not arg.startswith("--") :
                        raise SyntaxError("Terrible Commend arg format")
                    arg_name = arg[2:].strip()
                else:
                    arg_value= arg.strip()
                    if arg_name:
                        cmd_arg[arg_name] = arg_value
                    else:
                        raise SyntaxError("Terrible Commend arg format")

        return cmd_arg

    
    def __getitem__(self,item):
        """build the iterator, to access the args easily

        Params:
           item (str) : target arguments/property
        
        return: related value
        """
        if not isinstance(item,str):
            raise TypeError("index must be a str!")
        if item in self.cmd_arg:
            param=self.cmd_arg[item]
        elif item in self.lib_arg:
            param=self.lib_arg[item]
        elif item in self.alg_arg:
            param=self.alg_arg[item]
        else:
            raise KeyError('Not find the parameter %s'%item)


        ## convert param from str to different types value
        try:
            value=eval(param) # get the ture type data(int/float/list) from str 
            if not isinstance(value,(str,int,float,list,tuple,bool,None.__class__)):
                value=param
        except:
            if param.lower()=='true':
                value=True
            elif param.lower()=='false':
                value=false
            else:
                value=param
        return value

    def __getattr__(self,item):
        return self[item]

    def __str__(self):
        lib_info = '\n'.join(["{}={}".format(arg, value) for arg, value in self.lib_arg.items()])
        alg_info = '\n'.join(["{}={}".format(arg, value) for arg, value in self.alg_arg.items()])
        info = "\n datasets hyperparameters:\n%s\n\n%s's hyperparameters:\n%s\n" % (lib_info, self["recommender"], alg_info)
        return info


        