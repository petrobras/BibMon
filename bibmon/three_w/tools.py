import configparser
import importlib.resources as pkg_resources
import os


def parse_dataset_ini(dataset_ini_path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(dataset_ini_path)
    
    return config