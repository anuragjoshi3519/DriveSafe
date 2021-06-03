from flask import Flask, render_template, request, Response, url_for
from drowsiness_detection import drive

app = Flask(__name__)
selected = False
start_score = None
stop_score = None

@app.route('/',methods=['GET','POST'])
@app.route('/home',methods=['GET','POST'])
def home():
    global start_score, stop_score
    if request.method=='POST':
            place = request.form.get('place')
            time = request.form.get('time')
            if int(place)==0 or int(time)==0:
                selected=False
                return render_template('index.html',title='DriveSafe',Selected=selected)
            
            selected = True
            start_score = 26 - int(place) - int(time) 
            stop_score = start_score//2
            return render_template('index.html',title='DriveSafe',Selected=selected)
    else:
        return render_template('index.html',title='DriveSafe',Selected=False)

@app.route('/stream')
def stream():
    return Response(drive(start_score,stop_score), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)