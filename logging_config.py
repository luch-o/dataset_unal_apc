LOGGING_CONFIG = {
    'version': 1, # required
    'disable_existing_loggers': True, # this config overrides all other loggers
    'formatters': {
        'simple': {
            'format': '%(asctime)s %(levelname)s -- %(message)s'
        },
        'whenAndWhere': {
            'format': '[%(asctime)s] %(levelname)s -- %(module)s.%(funcName)s:%(lineno)d -- %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        'file':{
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'formatter': 'whenAndWhere',
            'filename': 'logs/segmentation.log',
            'mode': 'w'
        }
    },
    'loggers': {
        '': { # 'root' logger
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        }
    }
}

LOGGING_CONFIG_POSTPROC = {
    'version': 1, # required
    'disable_existing_loggers': True, # this config overrides all other loggers
    'formatters': {
        'simple': {
            'format': '%(asctime)s %(levelname)s -- %(message)s'
        },
        'whenAndWhere': {
            'format': '[%(asctime)s] %(levelname)s -- %(lineno)d -- %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        'file':{
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'formatter': 'whenAndWhere',
            'filename': 'logs/mask_postproc.log',
            'mode': 'w'
        }
    },
    'loggers': {
        '': { # 'root' logger
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        }
    }
}

