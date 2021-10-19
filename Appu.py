from flask import Flask,render_template,Response,request,jsonify
import cv2
import numpy as np
import mediapipe as mp
import os
from keras.models import load_model
from deep_translator import GoogleTranslator
from gtts import gTTS


app=Flask(__name__)

#Global variables
global model
cap=cv2.VideoCapture(0)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_styled_landmarks(image, results):
    # Drawing face connections - styled
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(205,205,205), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(185,185,185), thickness=1, circle_radius=1)
                             ) 
    # Drawing pose connections - styled
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(80,44,121),thickness=1, circle_radius=1)
                             ) 
    # Drawing left-hand connections - styled
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2), 
                              mp_drawing.DrawingSpec(color=(121,44,250),thickness=2, circle_radius=2)
                             ) 
    # Drawing right-hand connections - styled 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2), 
                              mp_drawing.DrawingSpec(color=(121,44,250),thickness=2, circle_radius=2)
                             )
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


# load the pre-trained Keras model
model = load_model(r"D:\MP\action.h5")
        

@app.route('/')
def index():
    return render_template('JavascriptWebCam.html')


@app.route('/get_lang', methods=['POST'])
def get_lang():    
    data = request.get_json()
    global language 
    language = data['lang_code']
    return jsonify(language)


# left cam part
@app.route('/left_cam', methods=['GET', 'POST'])
def testfn():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')



def gen_frames():
    global frame
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            success, buffer = cv2.imencode('.jpg', frame)
            frame1 = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')

#rigth part
@app.route('/output', methods=['GET', 'POST'])
def outputfn():
    # GET request
    print("in output fn")
    #if request.method == 'GET':
    #message = {}
    txt = frame_predicit()
    for i in txt:
        print(i)
        message = {"output":i}
        print(message)
        #return jsonify(message)  # serialize and use JSON headers
        return Response(i,mimetype='text/html')

def frame_predicit():
    global frame
    sequence = []
    sentence = []
    predictions = []
    l = []
    remember_pred = ""
    threshold = 0.8
    actions = np.array(['Hello','this is','a demo', 'Project'])
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            sentence = []
            image, results = mediapipe_detection(frame, holistic)
        
            draw_styled_landmarks(image, results)
        
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
        
            if len(sequence) == 30:
                sentence = []
                l = []
                res = model.predict(np.expand_dims(sequence, axis=0))[0] 
                predictions.append(np.argmax(res))
            
                if np.unique(predictions[-20:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)]) 
                        else:
                            sentence.append(actions[np.argmax(res)])        
            
            
            if len(sentence) > 0 and remember_pred != sentence[0]:
                remember_pred = sentence[0]
              
                if(language == "Tamil"):
                    lang_code = 'ta'
                elif(language == "Hindi"):
                    lang_code = 'hi'
                elif(language == "Telugu"):
                    lang_code = 'te'
                elif(language == "Malayalam"):
                    lang_code = 'ml'
                else:  
                    lang_code = 'en'

                translated = GoogleTranslator(source='auto', target=lang_code).translate(sentence[0])
                target = lang_code
                speak = gTTS(text=translated, lang=target, slow = False)
                speak.save("captured_voice.mp3")    
            
                os.system("captured_voice.mp3")
                global text_output
                text_output = sentence[0]
                l.append(text_output)
                sentence = []
                if text_output != "": 
                    yield text_output
            
            if cv2.waitKey(10) & 0xFF == ord('q'): 
                break
        cv2.destroyAllWindows()
        cap.release()

if __name__ == "__main__":
    app.run(debug=True)