from configparser import ConfigParser
from pathlib import Path


CONFIG_FILE = Path('aurore') / 'credentials.ini'

def config(file_name: CONFIG_FILE, section: str = 'hugging_face') -> dict:
    """ Retrouve les paramètres de connexions du compte Hugging face
    depuis le fichier credentials.ini """
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(file_name)

    # prendre le nom de la config, par défaut : hugging_face
    credentials = {}
    if parser.has_section(section):
        params = parser.items(section)
        if params :
            for param in params:
                credentials[param[0]] = param[1]
        else :
            raise Exception('Vous n\'avez pas renseigné vos credentials dans le fichier de configuration.')
    else:
        raise Exception(
            f'L\'entrée {section} n\'a pas été trouvée dans le fichier {file_name}. Avez vous renseigné vos credentials ? ')

    return credentials