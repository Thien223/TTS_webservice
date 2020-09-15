from waitress import serve
import app
import os
## set current working directory to source code directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("\nCurrent working directory: ---'{}'---\n".format(os.getcwd()))



if __name__=='__main__':
    serve(app.app, host='0.0.0.0', port=1000)