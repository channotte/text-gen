# Le réseau de neurones qui écrivait des romans

Bonjour et bienvenues dans ce codelab ! 
Le but de ce projet est de vous faire découvrir ou redécouvrir la data science à partir d'un cas pratique.
Nous allons entrainer un réseau auteur de bout en bout qui aura la plume de George Sand. 


# PRE-REQUIS :

#### Installer Docker et git

#### Cloner le projet

#### Se créer un compte Hugging Face

#### Générer un token d'écriture sur Hugging Face.

1. Pour cela, aller sur votre compte Hugging face > Settings > Access Tokens > New token 
2. Copiez ce token
3. Collez le dans le fichier aurore/credentials.ini après le TOKEN=
4. Dans la ligne HUGGING_FACE_PSEUDO renseignez après le '=' votre pseudo Hugging face

#### Construire l'image docker 

`docker build -t text-gen .`

#### Pour lancer l'application web : 
`docker run -v $PWD/aurore:/aurore aurore python aurore/5_flask.py`

et ouvrez l'une des deux urls suivants :

+ http://172.17.0.2/generate : pour utiliser le réseau de neurones
+ http://172.17.0.2/ : pour avoir des informations sur le projet Aurore

#### Codez !

Bon courage à tous. 

En cas de soucis, n'hésitez pas à nous solliciter !
