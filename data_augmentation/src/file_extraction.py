import tarfile
import os

def extract_tgz(tgz_path, output_dir):
    """
    Extrait tous les fichiers d'une archive .tgz dans le dossier spécifié.

    :param tgz_path: Chemin vers le fichier .tgz
    :param output_dir: Répertoire de destination
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
        print(f"Fichiers extraits dans : {output_dir}")

# Exemple d'utilisation
tgz_file = r"..\data\raw\OpenSeizureDatabase_V1.8.tgz"
output_folder = "dossier_extrait"
extract_tgz(tgz_file, output_folder)
