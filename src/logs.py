import logging
import os

class Logger():
    
    """
    A class to create a logger object to log model, training, and evaluation information.
    """
    
    def __init__(self, log_dir='../logs/', log_file_name='log.log'):
        
        """
        Args:
        log_dir (str): The directory to save the log file.
        log_file_name (str): The name of the log file.
        """
    
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, log_file_name)
        
        logging.basicConfig(filename=self.log_file, 
                            level=logging.INFO, 
                            format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        
        self.logger = logging.getLogger('logger')
        