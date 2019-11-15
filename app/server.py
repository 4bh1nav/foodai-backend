from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai.vision import *

path = Path(__file__).parent
app = Starlette()
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    #await download_file(model_file_url, path/'models'/f'{model_file_name}.pkl')
    learn = load_learner(path/'models')
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    cat,index,preds = learn.predict(img)
    return JSONResponse({'result': str(cat)})


def top_5_pred_labels(preds, classes):
    top_5 = np.flip(np.argsort(preds.numpy()))[:5]
    labels = []
    for i in range(len(top_5)):
        labels.append(classes[top_5[i]])
    return labels

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)

