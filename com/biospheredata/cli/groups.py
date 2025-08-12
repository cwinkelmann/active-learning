import json
from pathlib import Path

import click
from loguru import logger

@click.group()
# @click.command()
def cli():
    """Biospheredata command line utility

    \b
    * generate a usable orthomosaic with
    * Generate training data from corrected annotations (hasty or yolo) by slicing, train/test split and config data generation
    * candidate proposals to be corrected by an annotation tool

    Examples

    \b
    This is
    a paragraph
    without rewrapping.

    And this is a paragraph
    that will be rewrapped again.
    """
    pass

@cli.group()
def mission():
    """
    Interact with missions


    @param action:
    @param name:
    @return:
    """

@cli.group()
def expedition():
    """
    Interact with missions
    @return:
    """


@cli.group()
def orthophoto():
    """
    Interact with orthophotos


    @param action:
    @param name:
    @return:
    """

@cli.group()
def zooniverse():
    """
    process zooniverse data dumps


    @param action:
    @param name:
    @return:
    """