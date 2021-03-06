from fastapi import FastAPI
import joblib
import uvicorn
import pandas as pd

app = FastAPI()
model = joblib.load("modelnew.pkl")


@app.get('/')
def index():
    return {'message': 'Hello world Use /predict/ to get endpoint and pass 50 answeres in format of arr[1,2,3...] or srting 1234'}

@app.get('/predict/{data}')
def predict_cluster(data):
    if len(data)==50:
        
        if type(data)==str:
            # df = processor(data)
            dfx=[]
            for ix in data:
                dfx.append(int(ix))
    #         print(dfx)
            df = pd.DataFrame(dfx)
            df = df.T
            cluster = model.predict(df)
            result = str(int(cluster))
            return{'Cluster': result}


        elif type(data)==list:
            df = pd.DataFrame(data)
            df = df.T
            cluster = model.predict(df)
            result = str(int(cluster))
            return{'Cluster': result}
    else:
        return {'message': 'Expected 50 Answer, got either more or less'}


# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port = 8000)


