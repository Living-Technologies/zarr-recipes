#!/usr/bin/env python

from flask import Flask, Response, send_from_directory, send_file
import pathlib
import markdown
import sys
import io
import zipfile

app = Flask(__name__)

zarrZip = None
rootPath = None
zarrName = None

winSlashes = False
@app.route('/<path:path>')
def mark_down( path ):
    if zarrZip is None:
        return send_from_directory(rootPath, path)
    else:
        try:
            fullpath = "%s/%s"%(zarrName, path)
            print(fullpath)
            if winSlashes:
                fullpath = fullpath.replace("/", "\\")
            print(fullpath)
            data = zarrZip.open( fullpath )
            print(data)
            name = path.split("/")[-1]
            return send_file(
                     data,
                     download_name='name',
                     mimetype='application/octet-stream'
               )
        except:
            return Response(response = "no File!", status=404)

    #

@app.route("/favicon.ico")
def favicon():
    return Response( response = open("favicon.png", 'rb').read(), mimetype="image/png" )



if __name__=="__main__":
    src = pathlib.Path(sys.argv[1])
    zarrName = src.name
    if zarrName.endswith("zip"):
        zarrZip = zipfile.ZipFile(sys.argv[1], 'r')
        zarrName = zarrName.replace(".zip", ".zarr")
        info = zarrZip.filelist[0]
        if "\\" in info.filename:
            winSlashes = True
    else:
        rootPath = src.absolute()
    app.run(port=5050)
