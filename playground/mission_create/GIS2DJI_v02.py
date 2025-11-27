# -*- coding: utf-8 -*-
"""
Created in December 2023

@author: Nicolas Cadieux
njacadieux.gitlab@gmail.com
https://gitlab.com/njacadieux
https://www.youtube.com/@nicolascadieux5881


GPL-3.0-or-later
This program is free software: you can redistribute it and/or modify it under
the terms of theGNU General Public License as published by the Free Software
Foundation, either version 3 of theLicense, or (at your option) any later
version.This program is distributed in the hope that it
will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.You should have received a copy of the GNU
General Public License along with this program.
If not, see <https://www.gnu.org/licenses/>.

"""
import os
import sys
import geopandas as gpd
import simplekml
import fiona
from shapely import Polygon, LineString, Point, MultiPolygon,\
     MultiLineString, MultiPoint
from fiona.drvsupport import supported_drivers
#  Load support for kml read libraries from fiona
supported_drivers['LIBKML'] = 'rw'
import tkinter
import tkinter.messagebox
import customtkinter
from tkinter import ttk, Menu, filedialog, StringVar

software_version ='GIS2DJI_V02'
# set customtkinter appearance
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# =============================================================================
#                          USER VARIABLES
# =============================================================================

# INPUT_PATH = r''  # input path to all files
# NAME = 'name'
# NAME = None  # field name for 'name' in input .shp
# DESCRIPTION = 'desc'  # field name for 'description' in input .shp

# =============================================================================
#                      OPTIONAL USER VARIABLES
# =============================================================================

# OUTPUT_PATH = os.path.join(INPUT_PATH, 'DJI_kml_output')
SCANDIRECTORY_AND_SUBDIRECTORIES = 'no'  # no or yes
# EXTENSION_FILTRE = ['gpkg', 'shp',
                    # 'mif', 'tab', 'geojson',
                    # 'gml', 'kml', 'kmz']  # search for all shape files
# STR_FILE_FILTER = ''  # filter for input files name ex: 'Rigaud'= Rigaud_1.shp

# =============================================================================
#                         FUNCTIONS
# =============================================================================

def toto(string):
    print ('print from function ' + string)

def create_directory(path: str):
    """Create a directory. Will pass if directory has been added
      by another tread."""
    # err = ''
    if os.path.isdir(path):
        pass
    else:
        try:
            os.makedirs(path)
        except WindowsError as err:
            pass
            return err

def scan_directory(path: str, extensions_lts_flt='', str_flt=''):
    # new add multiple file extensions.  Not just one.
    file_lst = []
    for (directory, sub_directory, file_name) in os.walk(path):
        for files in file_name:
            for q in extensions_lts_flt:
                if files.lower().endswith(q) and str_flt in\
                        files:
                    file_filter = os.path.join(directory, files)
                    file_lst.append(file_filter)
        if SCANDIRECTORY_AND_SUBDIRECTORIES == 'no':
            break  # this break will exit the loop and limite to 1 directory
        elif SCANDIRECTORY_AND_SUBDIRECTORIES == 'yes':
            pass
    return file_lst

# =============================================================================
#     Main functions
# =============================================================================

def export_pg(geom, ofn, index, output_path):
    err =[]
    ext = list(geom.exterior.coords)
    int_ring = []
    for interior in geom.interiors:
        int_ring.append(list(interior.coords))
    kml = simplekml.Kml()
    obj = kml.newpolygon(name=ofn,
                         description=index)
    obj.outerboundaryis = ext
    if int_ring == []:
        pass
    else:
        obj.innerboundaryis = int_ring
    output_file_name = os.path.join(output_path,
                                    ofn + '-' + str(index) + '-' +
                                    geom.geom_type + '.kml')
    n = 1
    while os.path.isfile(output_file_name):
        output_file_name = os.path.join(output_path,
                                    ofn + '-' + str(index) + '-' +
                                    geom.geom_type +'('+ str(n) +')'+ '.kml')
        n += 1

    kml.save(output_file_name)
    err#.append('Exporting: '+ output_file_name)
    return(err)

def export_line(geom, ofn, index, output_path):
    err = []
    xyz = list(geom.coords)
    kml = simplekml.Kml()
    obj = kml.newlinestring(name=ofn,
                            description=index)
    obj.coords = xyz
    output_file_name = os.path.join(output_path,
                                    ofn + '-' + str(index) + '-' +
                                    geom.geom_type + '.kml')
    n = 1
    while os.path.isfile(output_file_name):
        output_file_name = os.path.join(output_path,
                                    ofn + '-' + str(index) + '-' +
                                    geom.geom_type +'('+ str(n) +')'+ '.kml')
        n += 1
    
    kml.save(output_file_name)
    err.append('Exporting: '+ output_file_name)
    return(err)


def export_point(geom, ofn, index, output_path):

    # This will export the point as a kmz.  According to the DJI api
    # documentation, points need to be kmz.
    err = []
    xyz = list(geom.coords)
    kml = simplekml.Kml()
    obj = kml.newpoint(name=ofn,
                       description=index)
    obj.coords = xyz
    output_file_name = os.path.join(output_path,
                                    ofn + '-' + str(index) + '-' +
                                    geom.geom_type + '.kmz')
    n = 1
    while os.path.isfile(output_file_name):
        output_file_name = os.path.join(output_path,
                                        ofn + '-' + str(index) + '-' +
                                        geom.geom_type +'('+ str(n) +')'+'.kmz')
        n+=1

    kml.savekmz(output_file_name)
    err.append('Exporting: '+ output_file_name)

    # Create PseudoWayPoint. This is a zero lenght line
    kml = simplekml.Kml()
    xyz.append(xyz[0])
    obj = kml.newlinestring(name=ofn,
                            description=index)
    obj.coords = xyz
    output_file_name = os.path.join(output_path,
                                    ofn + '-' + str(index) + '-' +
                                    'PseudoPoint' + '.kml')
    n = 1
    while os.path.isfile(output_file_name):
        output_file_name = os.path.join(output_path,
                                        ofn + '-' + str(index) + '-' +
                                        'PseudoPoint' +'('+ str(n) +')'+ '.kml')
        n+=1
        
        
    kml.save(output_file_name)
    err.append('Exporting: '+ output_file_name)
    return(err)


def main_function(file: str, output_path):
    msg = []
    fn = os.path.basename(os.path.splitext(file)[0])  # toto.shp --> toto
    fnx = (os.path.basename(os.path.splitext(file)[1])).strip('.')  # .shp
    ffn = os.path.basename(file)  # toto.shp
    #  Load support for kml read libraries from fiona
    # gpd.io.file.fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
    msg.append('Reading:'+ffn)
    try:
        layerlist = fiona.listlayers(file)
    except:
        msg.append('An exception occurred.'+' Cannot read'+str(file))
        msg.append('Skipping file!')
        return

    cnt = 0
    for layers in layerlist:
        layer_name = layerlist[cnt]
        cnt += 1
        try:
            layers = gpd.read_file(file, layer=layers)
        except:
            msg.append('An exception occurred. Cannot not read layer '+layer_name)
            msg.append('Skipping layer.')
            continue
        if len(layerlist) == 1:
            if fn == layer_name:
                ofn = fn + '-' + fnx
            elif fn != layer_name:
                ofn = fn + '-' + fnx + '-' + layer_name
        elif len(layerlist) > 1:
            ofn = fn + '-' + fnx + '-' + layer_name

        # Check CRS and reproject if needed
        crs = layers.crs
        # TODO Test if I need to specify CRS
        if crs is None:
            msg.append(layer_name+ ' has no CRS (Coordinate Reference System). Output .kml will not be well georeferenced')
            msg.append('if the input file is not already in EPSG 4326 (WGS84).')
            msg.append('Resulting file may not be valid and may not open properly in drone software.')
            tkinter.messagebox.showinfo(title='Warning!!!', message = str(msg))

        elif crs == 'epsg:4326':
            # print('CRS for', layer_name, 'layer is EPSG 4326')
            pass
        else:
            msg.append(layer_name+' layer has been reprojected to EPSG 4326.')
            layers = layers.to_crs(4326)

        # Iterate each row of each layers and create individual kml files
        for index, row in layers.iterrows():
            geom = (row.geometry)

            # Check if geom is valid, try to fix
            if geom.is_valid is True:
                pass
            else:
                msg.append('Invalid geometry found: Layer name = '+ '"'+
                      str(layer_name)+ '"'+ 'index = '+ str(index))
                msg.append('We will try to fix the geometry...')
                gps = (gpd.GeoSeries(geom))
                hope4thebest = gps.make_valid()
                geom = (hope4thebest[0])
                msg.append('Geometry is probably fixed. Please verify.')

            if geom.geom_type == 'Polygon' or geom.geom_type == 'MultiPolygon':
                if geom.geom_type == 'Polygon':
                    err = export_pg(geom, ofn, index, output_path)
                    for x in err:
                        msg.append(x)
                    
                elif geom.geom_type == 'MultiPolygon':
                    for parts in range(len(geom.geoms)):
                        # print (geom.geoms[parts])
                        # print (parts)
                        index_part = str(index) + '-' + str(parts)
                        err = export_pg(geom.geoms[parts], ofn, index_part, output_path)
                        for x in err:
                            msg.append(x)

            elif geom.geom_type == 'LineString' or \
                    geom.geom_type == 'MultiLineString':

                if geom.geom_type == 'LineString':
                    err = export_line(geom, ofn, index, output_path)
                    for x in err:
                        msg.append(x)
                    
                elif geom.geom_type == 'MultiLineString':
                    for parts in range(len(geom.geoms)):
                        # print (geom.geoms[parts])
                        # print (parts)
                        index_part = str(index) + '-' + str(parts)
                        err = export_line(geom.geoms[parts], ofn, index_part, output_path)
                        for x in err:
                            msg.append(x)
                            
            elif geom.geom_type == 'Point' or geom.geom_type == 'MultiPoint':
                if geom.geom_type == 'Point':
                    err = export_point(geom, ofn, index, output_path)
                    for x in err:
                        msg.append(x)
                    
                elif geom.geom_type == 'MultiPoint':
                    for parts in range(len(geom.geoms)):
                        # print (geom.geoms[parts])
                        # print (parts)
                        index_part = str(index) + '-' + str(parts)
                        err = export_point(geom.geoms[parts], ofn, index_part, output_path)
                        for x in err:
                            msg.append(x)
            else:
                msg.append(geom.geom_type+
                      ' is not supported. Geometry not exported')
                continue
    return(msg)

# =============================================================================
#         # configure GUI
# =============================================================================

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

# =============================================================================
#         # configure window
# =============================================================================
        self.title("GIS 2 DJI")
        self.geometry(f"{1200}x{580}")

# =============================================================================
#  tk variables
# =============================================================================
        self.input_folder = StringVar()
        self.output_folder = StringVar()
        self.select_extention = StringVar() # use toto = list(eval(self.file_lst.get())) ONLY WORKS WITH('','')2
        self.file_lst = StringVar() # use toto = list(eval(self.file_lst.get())) ONLY WORKS WITH('','')2

# =============================================================================
#         # configure grid layout (4x4)
# =============================================================================
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

# =============================================================================
#         #Menu bar
# =============================================================================
        self.menubar = Menu(self)
        self.filemenu = Menu(self.menubar, tearoff=0)
        # filemenu.add_command(label="New", command=donothing)
        self.filemenu.add_command(label="Import", command=self.get_input_files_path)
        self.filemenu.add_command(label="Export", command=self.get_output_folder_path)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.destroy)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.helpmenu = Menu(self.menubar, tearoff=0)
        self.helpmenu.add_command(label="Help Index", command=self.help)
        self.helpmenu.add_command(label="About...", command=self.about)
        self.menubar.add_cascade(label="Help", menu=self.helpmenu)
        self.config(menu=self.menubar)

# =============================================================================
#         # create sidebar BATCH FILE MODE frame with widgets
# =============================================================================
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Batch mode", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.optionmenu_extensions_var = customtkinter.StringVar(value="*.*")
        self.optionmenu_extensions = customtkinter.CTkOptionMenu(self.sidebar_frame, dynamic_resizing=False,
                                                        values=['*.*', 'shp', 'kml', 'kmz', 'gpkg', 'geojson', 'gml', 'mif', 'tab'],
                                                        variable=self.optionmenu_extensions_var, command=self.optionmenu_callback)
        
        self.optionmenu_extensions.grid(row=1, column=0, padx=20, pady=10)

        self.entry = customtkinter.CTkEntry(self.sidebar_frame, placeholder_text="txt filter...")
        self.entry.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.sidebar_button_input_dir = customtkinter.CTkButton(self.sidebar_frame, text='Input directory', command=self.get_input_folder_path)
        self.sidebar_button_input_dir.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_output_dir = customtkinter.CTkButton(self.sidebar_frame, text='Output directory', command=self.get_output_folder_path)
        self.sidebar_button_output_dir.grid(row=4, column=0, padx=20, pady=10)

        self.sep1 = ttk.Separator(self.sidebar_frame, orient = 'vertical')
        self.sep1.grid(row=5, column=0, padx=20, pady=10, sticky="ew")

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 20))

# =============================================================================
#         # create main entry and button
# =============================================================================
        self.main_button_1 = customtkinter.CTkButton(master=self,
                                                     fg_color="transparent",
                                                     border_width=2,
                                                     text_color=("gray10",
                                                                 "#DCE4EE"),
                                                     text='Process files...',
                                                     command=self.process_files)
        self.main_button_1.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")
# =============================================================================
#         # create textbox
# =============================================================================
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=0, rowspan=3, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")


        
    def help(self):
        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", 'Usage, batch mode: \n' +
                            '1: Start by selecting a file format if needed.\n' +
                            '2: Select a string filter if needed.\n'+
                            '3: Select an input then an output directory. They must be different directories.\n'+
                            '\t*** WARNING *** Output files with identical file names will be overwritten!!!\n'+
                            '4: Press Process files button.\n\n\n')
        
        self.textbox.insert("end", 'Usage, single file mode: \n' +
                            '1: Go to file --> Import --> Select one or multiple files.\n'+
                            '\tYou can select a file format.  Live dangerously will show you all files, some formats may be compatible but have not been tested.\n'+
                            '3: Go to file --> Export to select an output directory.  Must be different than the input file directory.\n'+
                            '\t*** WARNING *** Output files with identical file names will be overwritten!!!\n'+
                            '\t Output files with identical file names will be overwritten!!!\n'+
                            '4: Press "Process files" button.\n\n\n')
        
        self.textbox.insert("end", 'GIS_2_DJI has been tested with the following file formats: gpkg, shp, mif, tab, geojson, gml, kml and kmz.\n' +
                            'GIS_2_DJI will scan every file, every layer and every geometry collection\n' + 
                            '(ie: MultiPoints) and create one output kml or kmz for each object found.\n' +
                            'Output file naming convention is: shp2dji-gpkg-MP-0-1-Point.kmz\n' +
                            '\t-Original file name --> "shp2dji"\n' +
                            '\t-Original file extension --> "gpkg"\n' +
                            '\t-Original layer (if file has layers) --> "MP"\n' +
                            '\t-Source index number --> "0"\n'+
                            '\t-Source multigeometry index (if is a multigeometry) --> "1"\n' +
                            '\t-Geometry type --> "Point"\n' +
                            '\t-.kml or kmz --> ".kmz"\n\n' +
                            'It will import points, lines and polygons, and converted each object into a compatible DJI kml file.\n' +
                            'Lines and polygons will be exported as kml files.  Points will be converted as PseudoPoints.kml.\n' +
                            'A PseudoPoints fools DJI to import a point as it thinks it''s a line with 0 length.\n'+
                            'This allows you to import points in mapping missions. Points will also be exported as Point.kmz because PseudoPoints\n'+
                            'are not visible in a GIS or in Google Earth. The .kmz file format should make points compatible with some DJI mission software.')
    def about(self):
        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", 'Built by: \n' +
                            'Nicolas Cadieux: McGill ARSL \n' +
                            'Version: '+ software_version+'\n'+
                            'Please visite https://gitlab.com/njacadieux for latest version or bug reports.\n\n')
        self.textbox.insert("end", 'Changes in v02: \n' +
                            '-Checks to see if output file name exists.  Adds a file version number if file exists. \n' +
                            '-Warning to user if input file has no CRS')
            
    
        
    def donothing(self):
        # self.textbox.delete("0.0","end")
        self.textbox.insert("end", '\n Function in contruction...')

    def get_input_folder_path(self):
        self.input_folder_selected = filedialog.askdirectory()
        self.input_folder.set(self.input_folder_selected)
        if self.output_folder.get() == self.input_folder.get():
            tkinter.messagebox.showinfo(title='Warning!!!', message='Input and Output directory are the same.\n Please select different directories...')
        self.textbox.delete("0.0", "end")
        self.textbox.insert("end", "Input directory: " + self.input_folder.get() +'\n')

        self.selection = self.optionmenu_extensions.get()
        if self.selection == '*.*':
            self.selection = ('gpkg', 'shp',
                                'mif', 'tab', 'geojson',
                                'gml', 'kml', 'kmz')
            self.select_extention.set(self.selection)
        else:
            lst = []
            lst.append(self.selection)
            self.selection = lst
            self.select_extention.set(self.selection)

        self.fs = scan_directory(self.input_folder.get(), self.selection, self.entry.get())
        # use toto = list(eval(self.file_lst.get()))
        self.file_lst.set(self.fs)

        self.textbox.insert("end", 'The following files where found: \n\n')
        for f in self.fs:
            self.textbox.insert("end", '\t' + f + '\n')
        self.textbox.insert("end", '\n Please select an empty Output directory. \n' +
                            '**** WARNING:  Files in the output directory will ' +
                            'be overwritten. ****\n')

    def get_output_folder_path(self):
        self.ouput_folder_selected = filedialog.askdirectory()
        self.output_folder.set(self.ouput_folder_selected)
        if self.output_folder.get() == self.input_folder.get():
            tkinter.messagebox.showinfo(title='Warning!!!', message='Input and Output directory are the same.\n Please select different directories...')
        # https://stackoverflow.com/questions/51877124/how-to-select-a-directory-and-store-it-into-a-variable-in-tkinter
        self.textbox.insert("end", "Output directory: " + self.output_folder.get() +'\n')

    def get_input_files_path(self):
        self.input_files = filedialog.askopenfilenames(filetypes=(
            ("All compatible files",('*.gpkg', '*.shp', '*.mif', '*.tab',
                                     '*.geojson','*.gml', '*.kml', '*.kmz')),
            ("Geopackage", '*.gpkg'),
            ('ShapeFiles','*.shp'),
            ('MapInfo mif','*.mif'),
            ('MapInfo tab','*.tab'),
            ('geojson', '*.geojson'),
            ('Geography Markup Language','*.gml'),
            ('Keyhole Markup Language','*.kml'),
            ('keyhole Markup Language','*.kmz'),
            ('Live Dangerously...','*.*'),
            ))
        self.input_folder.set(os.path.dirname((self.input_files[0])))
        if self.output_folder.get() == self.input_folder.get():
            tkinter.messagebox.showinfo(title='Warning!!!', message='Input and Output directory are the same.\n Please select different directories...')
        self.file_lst.set(self.input_files)
        self.textbox.delete("0.0", "end")
        self.textbox.insert("end", 'The following files where found: \n\n')
        for f in self.input_files:
            self.textbox.insert("end", '\t' + f + '\n')
        self.textbox.insert("end", '\n Please select an empty Output directory. \n' +
                            '**** WARNING:  Files in the output directory will ' +
                            'be overwritten. ****\n')

    def optionmenu_callback(self, choice):
        # print("optionmenu dropdown clicked:", choice)
        if choice == '*.*':
            choice = ['gpkg', 'shp',
                                'mif', 'tab', 'geojson',
                                'gml', 'kml', 'kmz']
            self.f_filter = choice
        else:
            self.f_filter = [choice]
        print("optionmenu dropdown clicked:", choice)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

# =============================================================================
#         # set default values
# =============================================================================
        self.appearance_mode_optionemenu.set("System")
        self.scaling_optionemenu.set("100%")
        self.optionmenu_extensions.set("*.*")
        self.textbox.insert("0.0", 'Usage, batch mode: \n' +
                            '1: Start by selecting a file format if needed.\n' +
                            '2: Select a string filter if needed.\n'+
                            '3: Select an input then an output directory. They must be different directories.\n'+
                            '\t*** WARNING *** Output files with identical file names will be overwritten!!!\n'+
                            '4: Press Process files button.\n'+
                            'Select "Help" for more details...')
        # self.select_extention.set(('toto','toto'))
        
    def process_files(self):
        
        try:
            # print ('print from classe ' + self.folder_selected)
            # toto(self.input_folder.get())
            # toto(self.output_folder.get())
            # toto(self.select_extention.get())
            # toto(self.file_lst.get())
            go = 1
            self.ifget=self.input_folder.get()
            if self.ifget=='':
                # print ('Please select input folder or file.')
                self.textbox.insert("end", '**** Error!!! ****\n')
                self.textbox.insert("end", 'Please select an input directory or file\n\n')
                go = 0
            else:
                # print ('Selected input folder:', self.ifget)
                pass

            self.ofget=self.output_folder.get()
            if self.ofget=='':
                # print('Please select output folder')
                self.textbox.insert("end", '**** Error!!! ****\n')
                self.textbox.insert("end", 'Please select an output directory\n\n')
                go = 0
            else:
                # print('Selected output folder:', self.ofget)
                pass
            if self.ifget == self.ofget:
                self.textbox.insert("end", '**** Error!!! ****\n')
                self.textbox.insert("end", 'Input and output directories MUST be different.\n')
                self.textbox.insert("end", 'Current Input directory is:'+str(self.ifget) + '\n')
                self.textbox.insert("end", 'Current Output directory is:'+str(self.ofget) + '\n')
                go = 0

            # Error for gui variable list vs str
            try:
                # if a list
                self.filelstget = (list(eval(self.file_lst.get())))
                # print(list(eval(self.file_lst.get())))
                # print('Importing the following files:\n')
                # for self.f in self.filelstget:
                #     print('\t', self.f)

            except Exception as ex:
                # if a string
                self.filelstget=self.file_lst.get()
                if self.filelstget == '':
                    self.textbox.insert("end", '**** Error!!! ****\n')
                    self.textbox.insert("end", 'No input files found. Please select input files.\n\n')
                    go = 0
                # print('Importing the following file:\n')
                # print('\t',self.filelstget)


            if go == 1:
                self.textbox.insert("end",'\n')
                self.textbox.insert("end",'Processing files...\n')
                for file in self.filelstget:
                    self.textbox.insert("end",'Importing:' + file + '\n')
                    msg = main_function(file, self.ofget)
                    for m in msg:
                        self.textbox.insert("end", '\t'+m+'\n')
                self.textbox.insert("end",'\n')        
                self.textbox.insert("end",'Done!  Have a safe flight!\n')

        except AttributeError as err:
            self.textbox.delete("0.0", "end")
            self.textbox.insert("0.0", 'Problem processing files... Please report bug to njacadieux.gitlab@gmail.com')
# =============================================================================
#                            End of GUI
# =============================================================================
        
if __name__ == "__main__":
    app = App()
    app.mainloop()
