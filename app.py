import os, json, random, math, io, uuid
from collections import deque
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, send_file, redirect, url_for, flash, render_template_string
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader
from openai import OpenAI

# ============ ИИ ============
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # на хостинге ключ хранится в переменных окружения
client = None; AI_AVAILABLE = False
if GROQ_API_KEY:
    try:
        client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        AI_AVAILABLE = True
    except: pass

# ============ ЗАДАЧИ ============
def generate_tasks_ai(topic, grade=7, count=16):
    prompt = f"""Ты — учитель математики. Создай ровно {count} задач. ТЕМА: {topic}, КЛАСС: {grade}.
Требования: разной сложности, один короткий числовой ответ (1-5 символов), текст до 2 строк, 3 неправильных ответа — типичные ошибки.
Верни СТРОГО JSON без markdown: {{"tasks": [{{"task_text":"...","draw_type":"text","draw_params":{{}},"correct_answer":"6.2","wrong_answers":["5.12","6.8","7.2"]}}]}}"""
    try:
        resp = client.chat.completions.create(model="llama-3.1-8b-instant",
            messages=[{"role":"system","content":"Отвечай ТОЛЬКО JSON."},{"role":"user","content":prompt}],
            temperature=0.8, max_tokens=4000)
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"): raw = "\n".join(raw.split("\n")[1:-1])
        data = json.loads(raw)
        tasks = data.get("tasks", data) if isinstance(data, dict) else data
        return [{"task_text":str(t.get("task_text","?")),"draw_type":t.get("draw_type","text"),
                 "draw_params":t.get("draw_params",{}),"correct_answer":str(t.get("correct_answer","?")),
                 "wrong_answers":[str(w) for w in t.get("wrong_answers",["?","??","???"])][:3]} for t in tasks[:count]]
    except Exception as e:
        print(f"ИИ ошибка: {e}")
        return generate_tasks_fallback(topic, count)

def generate_tasks_fallback(topic, count=16):
    tasks = []; tl = topic.lower()
    for _ in range(count):
        if "дроб" in tl:
            a=round(random.uniform(0.1,9.9),1); b=round(random.uniform(0.1,9.9),1); op=random.choice(["+","-","×"])
            if op=="+": ans=round(a+b,2); txt=f"{a} + {b} = ?"
            elif op=="-": a,b=max(a,b),min(a,b); ans=round(a-b,2); txt=f"{a} - {b} = ?"
            else: a=round(random.uniform(0.1,3),1); b=round(random.uniform(0.1,3),1); ans=round(a*b,2); txt=f"{a} × {b} = ?"
            tasks.append(_mk("text",txt,ans,_wrong(ans,True)))
        elif "уравнен" in tl:
            x=random.randint(1,20); a=random.randint(2,10); b=random.randint(1,50); c=a*x+b
            tasks.append(_mk("equation",f"{a}x+{b}={c}\nНайди x",x,_wrong(x),{"equation":f"{a}x + {b} = {c}"}))
        elif "угл" in tl or "градус" in tl:
            known=random.randint(20,160); ans=180-known
            tasks.append(_mk("text",f"Смежные углы:\n∠1={known}°, ∠2=?",f"{ans}°",[f"{w}°" for w in _wrong(ans)]))
        else:
            a=random.randint(2,50); b=random.randint(2,50); op=random.choice(["+","-","×"])
            if op=="+": ans=a+b; txt=f"{a} + {b} = ?"
            elif op=="-": a,b=max(a,b),min(a,b); ans=a-b; txt=f"{a} - {b} = ?"
            else: a=random.randint(2,12); b=random.randint(2,12); ans=a*b; txt=f"{a} × {b} = ?"
            tasks.append(_mk("text",txt,ans,_wrong(ans)))
    return tasks

def _mk(dt,txt,ans,wr,dp=None):
    return {"task_text":txt,"draw_type":dt,"draw_params":dp or {},"correct_answer":str(ans),"wrong_answers":[str(w) for w in wr[:3]]}
def _wrong(c, dec=False):
    s=set()
    for _ in range(50):
        if len(s)>=3: break
        if dec: w=round(c+random.choice([-1,-0.1,0.1,0.5,1])*random.randint(1,3),2)
        else: w=int(c)+random.choice([-5,-3,-1,1,2,3,5])
        if w!=c and w>0: s.add(w)
    while len(s)<3: s.add(int(c)+len(s)+1)
    return list(s)[:3]

# ============ ЛАБИРИНТ ============
class MathMaze:
    def __init__(self, cols=3, rows=6):
        self.cols=cols; self.rows=rows; self.total_cells=cols*rows
        self.neighbors={}
        for r in range(rows):
            for c in range(cols):
                idx=r*cols+c; nb=[]
                if c>0: nb.append(idx-1)
                if c<cols-1: nb.append(idx+1)
                if r>0: nb.append(idx-cols)
                if r<rows-1: nb.append(idx+cols)
                self.neighbors[idx]=nb
        self.correct_path=[]; self.corridors={}; self.cell_tasks={}
    def generate_path(self, start=0, end=None):
        if end is None: end=self.total_cells-1
        best=None
        for _ in range(30):
            path=[]; vis=set()
            def dfs(n):
                if n==end: path.append(n); return True
                vis.add(n); path.append(n); nb=self.neighbors[n][:]; random.shuffle(nb)
                for x in nb:
                    if x not in vis and dfs(x): return True
                path.pop(); vis.discard(n); return False
            if dfs(start) and (best is None or len(path)>len(best)): best=path[:]
        if best and len(best)>=4: self.correct_path=best
        else:
            from collections import deque as dq
            q=dq([(start,[start])]); vi={start}
            while q:
                n,p=q.popleft()
                if n==end: self.correct_path=p; break
                for x in self.neighbors[n]:
                    if x not in vi: vi.add(x); q.append((x,p+[x]))
        return self.correct_path
    def assign_tasks(self, tasks):
        path=self.correct_path; fin=path[-1]; idx=0
        for c in path[:-1]:
            if idx<len(tasks): self.cell_tasks[c]=tasks[idx]; idx+=1
        for c in range(self.total_cells):
            if c not in self.cell_tasks and c!=fin:
                self.cell_tasks[c]=tasks[idx] if idx<len(tasks) else random.choice(tasks)
                if idx<len(tasks): idx+=1
        self.corridors={}; edges=set()
        for i in range(len(path)-1):
            a,b=path[i],path[i+1]; e=(min(a,b),max(a,b)); edges.add(e)
            if a in self.cell_tasks: self.corridors[e]={"label":self.cell_tasks[a]["correct_answer"],"is_correct":True}
        for c in path[:-1]:
            wr=self.cell_tasks.get(c,{}).get("wrong_answers",[]); k=0
            for nb in self.neighbors[c]:
                e=(min(c,nb),max(c,nb))
                if e not in self.corridors and k<len(wr): self.corridors[e]={"label":wr[k],"is_correct":False}; k+=1
        return self

# ============ РИСОВАНИЕ ============
def draw_task_image(task):
    fig,ax=plt.subplots(figsize=(2.8,1.8),dpi=150); ax.axis('off')
    t=task.get("draw_type","text"); p=task.get("draw_params",{}); txt=task.get("task_text","")
    if t=="text": ax.text(0.5,0.5,txt,ha='center',va='center',fontsize=11,fontweight='bold',wrap=True,transform=ax.transAxes)
    elif t=="equation":
        eq=p.get("equation",txt); ax.text(0.5,0.82,"Найди x:",ha='center',va='center',fontsize=8,style='italic',transform=ax.transAxes)
        ax.text(0.5,0.4,eq,ha='center',va='center',fontsize=14,fontweight='bold',fontfamily='monospace',transform=ax.transAxes)
    elif t=="angle":
        rays=p.get("rays",[0,65,147]); labels=p.get("labels",["C","B","A"]); origin=(0.5,0.15); length=0.38
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        for i,a in enumerate(rays):
            rad=math.radians(a); ex=origin[0]+length*math.cos(rad); ey=origin[1]+length*math.sin(rad)
            ax.annotate("",xy=(ex,ey),xytext=origin,arrowprops=dict(arrowstyle="->",color='black',lw=1.5))
            if i<len(labels): ax.text(origin[0]+(length+0.07)*math.cos(rad),origin[1]+(length+0.07)*math.sin(rad),labels[i],fontsize=9,ha='center',va='center',fontweight='bold')
        ax.plot(*origin,'ko',markersize=4); ax.text(0.45,0.08,'O',fontsize=9,fontweight='bold'); ax.text(0.4,0.35,'x=?',fontsize=11,fontweight='bold',color='red')
    else: ax.text(0.5,0.5,txt,ha='center',va='center',fontsize=10,transform=ax.transAxes)
    plt.tight_layout(pad=0.3); buf=io.BytesIO()
    fig.savefig(buf,format='png',bbox_inches='tight',pad_inches=0.05,facecolor='white'); plt.close(fig); buf.seek(0); return buf

# ============ PDF ============
def build_maze_pdf(maze, title, filename):
    pw,ph=A4; c=pdf_canvas.Canvas(filename,pagesize=A4)
    ml=10*mm; mt=25*mm; gx=10*mm; gy=9*mm; cols=maze.cols; rows=maze.rows
    cw=min((pw-2*ml-(cols-1)*gx)/cols,58*mm); ch=min((ph-mt-15*mm-(rows-1)*gy)/rows,35*mm)
    sh=2*mm; rad=3*mm
    c.setFont("Helvetica-Bold",18); c.drawCentredString(pw/2,ph-18*mm,title.upper())
    pos={}; tw=cols*cw+(cols-1)*gx; sx=(pw-tw)/2
    for r in range(rows):
        for co in range(cols):
            idx=r*cols+co; pos[idx]=(sx+co*(cw+gx), ph-mt-(r+1)*ch-r*gy)
    for (a,b),info in maze.corridors.items():
        if a not in pos or b not in pos: continue
        xa,ya=pos[a]; xb,yb=pos[b]; lb=info["label"]; ca2,ra2=a%cols,a//cols; cb2,rb2=b%cols,b//cols
        if ra2==rb2:
            left=min(xa,xb)+cw; mid=ya+ch/2; h=ch*0.35
            c.setFillColorRGB(0.95,0.95,0.95); c.setStrokeColorRGB(0.3,0.3,0.3); c.setLineWidth(0.5)
            c.rect(left,mid-h/2,gx,h,fill=1,stroke=1); c.setFont("Helvetica-Bold",7); c.setFillColorRGB(0,0,0); c.drawCentredString(left+gx/2,mid-2,str(lb))
        elif ca2==cb2:
            mx=pos[min(a,b)][0]+cw/2; ty=pos[min(a,b)][1]; w=cw*0.3
            c.setFillColorRGB(0.95,0.95,0.95); c.setStrokeColorRGB(0.3,0.3,0.3); c.setLineWidth(0.5)
            c.rect(mx-w/2,ty-gy,w,gy,fill=1,stroke=1); c.setFont("Helvetica-Bold",7); c.setFillColorRGB(0,0,0); c.drawCentredString(mx,ty-gy/2-2,str(lb))
    for idx in range(maze.total_cells):
        if idx not in pos: continue
        x,y=pos[idx]
        c.setFillColorRGB(0.2,0.2,0.2); c.roundRect(x+sh,y-sh,cw,ch,rad,fill=1,stroke=0)
        c.setFillColorRGB(1,1,1); c.setStrokeColorRGB(0,0,0); c.setLineWidth(1.5); c.roundRect(x,y,cw,ch,rad,fill=1,stroke=1)
        if idx==maze.correct_path[0]:
            c.saveState(); c.setFont("Helvetica-Bold",7); c.setFillColorRGB(0.8,0,0); c.translate(x+4*mm,y+ch/2); c.rotate(90); c.drawCentredString(0,0,"СТАРТ"); c.restoreState()
            if idx in maze.cell_tasks: img=draw_task_image(maze.cell_tasks[idx]); c.drawImage(ImageReader(img),x+8*mm,y+2*mm,width=cw-10*mm,height=ch-4*mm,preserveAspectRatio=True,mask='auto')
        elif idx==maze.correct_path[-1]:
            c.setFont("Helvetica-Bold",18); c.setFillColorRGB(0,0.5,0); c.drawCentredString(x+cw/2,y+ch/2-4,"ФИНИШ")
        else:
            if idx in maze.cell_tasks: img=draw_task_image(maze.cell_tasks[idx]); c.drawImage(ImageReader(img),x+2*mm,y+1*mm,width=cw-4*mm,height=ch-2*mm,preserveAspectRatio=True,mask='auto')
    c.setFont("Helvetica",6); c.setFillColorRGB(0.5,0.5,0.5); c.drawCentredString(pw/2,5*mm,"Реши задачу -> найди ответ на коридоре -> иди дальше -> дойди до ФИНИША")
    c.showPage()
    c.setFont("Helvetica-Bold",16); c.setFillColorRGB(0,0,0); c.drawCentredString(pw/2,ph-20*mm,"ОТВЕТЫ (для учителя)")
    c.setFont("Helvetica",10); yp=ph-40*mm; c.drawString(15*mm,yp,f"Маршрут: {' -> '.join(map(str,maze.correct_path))}"); yp-=12*mm
    c.setFont("Helvetica",9)
    for cell in maze.correct_path[:-1]:
        t=maze.cell_tasks.get(cell,{}); txt=t.get("task_text","-").replace("\n"," "); ans=t.get("correct_answer","-")
        c.drawString(15*mm,yp,f"Ячейка {cell}: {txt}"); c.setFont("Helvetica-Bold",9); c.drawString(140*mm,yp,f"= {ans}"); c.setFont("Helvetica",9); yp-=8*mm
        if yp<20*mm: c.showPage(); yp=ph-20*mm
    c.showPage(); c.save(); return filename

# ============ FLASK САЙТ ============
site = Flask(__name__)
site.secret_key = "maze-key"
os.makedirs("output", exist_ok=True)

HTML_PAGE = r"""
<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Генератор лабиринтов</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);min-height:100vh;padding:20px}
.container{max-width:700px;margin:0 auto;background:#fff;border-radius:16px;padding:40px;box-shadow:0 20px 60px rgba(0,0,0,.3)}
header{text-align:center;margin-bottom:30px}h1{font-size:26px;color:#2c3e50;margin-bottom:6px}.sub{color:#7f8c8d;font-size:14px}
.fg{display:flex;flex-direction:column;gap:6px;margin-bottom:16px}label{font-weight:600;color:#34495e;font-size:14px}
input[type=text],select{padding:12px;border:2px solid #e0e0e0;border-radius:8px;font-size:15px;width:100%}input:focus,select:focus{border-color:#667eea;outline:none}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}.hint{font-size:12px;color:#999}.chk{display:flex;align-items:center;gap:8px}
.btn{padding:16px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border:none;border-radius:10px;font-size:18px;font-weight:700;cursor:pointer;width:100%;margin-top:10px}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(102,126,234,.4)}
.alert{padding:12px;border-radius:8px;margin-bottom:15px;background:#fdecea;color:#c0392b}
.tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}.tag{padding:6px 14px;background:#f0f0f0;border-radius:20px;font-size:13px;cursor:pointer}.tag:hover{background:#667eea;color:#fff}
.steps{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
.step{display:flex;align-items:center;gap:10px;padding:10px;background:#f8f9fa;border-radius:8px;font-size:13px;color:#555}
.num{width:28px;height:28px;background:#667eea;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0}
footer{margin-top:25px;text-align:center;color:#999;font-size:12px;border-top:1px solid #eee;padding-top:12px}
@media(max-width:600px){.row3{grid-template-columns:1fr}.steps{grid-template-columns:1fr}}
</style></head><body><div class="container">
<header><h1>🧩 Генератор математических лабиринтов</h1><p class="sub">Создайте лабиринт по любой теме — скачайте PDF для печати</p></header>
{% with m=get_flashed_messages() %}{% if m %}{% for msg in m %}<div class="alert">{{msg}}</div>{% endfor %}{% endif %}{% endwith %}
<form action="/generate" method="POST">
<div class="fg"><label>📚 Тема</label><input type="text" name="topic" placeholder="Например: десятичные дроби, углы, уравнения..." required><span class="hint">Любая тема школьной математики</span></div>
<div class="row3">
<div class="fg"><label>🎓 Класс</label><select name="grade">{% for i in range(1,12) %}<option value="{{i}}" {{'selected' if i==6}}>{{i}} класс</option>{% endfor %}</select></div>
<div class="fg"><label>📐 Столбцы</label><select name="cols"><option value="3" selected>3</option><option value="4">4</option></select></div>
<div class="fg"><label>📐 Строки</label><select name="rows"><option value="4">4</option><option value="5">5</option><option value="6" selected>6</option><option value="7">7</option></select></div>
</div>
<div class="fg"><label>✏️ Заголовок</label><input type="text" name="title" placeholder="Оставьте пустым — сгенерируется автоматически"></div>
<div class="fg chk"><label><input type="checkbox" name="use_ai" {% if ai %}checked{% else %}disabled{% endif %}> 🤖 Использовать ИИ</label>
{% if not ai %}<span class="hint">(ИИ не подключен)</span>{% endif %}</div>
<button class="btn" type="submit">🎲 Создать лабиринт и скачать PDF</button>
</form>
<div style="margin-top:25px;border-top:1px solid #eee;padding-top:15px"><b>💡 Примеры тем:</b><div class="tags">
<span class="tag" onclick="document.querySelector('[name=topic]').value='десятичные дроби'">Десятичные дроби</span>
<span class="tag" onclick="document.querySelector('[name=topic]').value='уравнения'">Уравнения</span>
<span class="tag" onclick="document.querySelector('[name=topic]').value='градусная мера углов'">Углы</span>
<span class="tag" onclick="document.querySelector('[name=topic]').value='обыкновенные дроби'">Обыкн. дроби</span>
<span class="tag" onclick="document.querySelector('[name=topic]').value='проценты'">Проценты</span>
<span class="tag" onclick="document.querySelector('[name=topic]').value='таблица умножения'">Умножение</span>
</div></div>
<div style="margin-top:20px;border-top:1px solid #eee;padding-top:15px"><b>📖 Как это работает:</b><div class="steps">
<div class="step"><div class="num">1</div> Введите тему</div><div class="step"><div class="num">2</div> ИИ создаёт задачи</div>
<div class="step"><div class="num">3</div> Строится лабиринт</div><div class="step"><div class="num">4</div> Скачайте PDF</div>
</div></div>
<footer>Генератор математических лабиринтов © 2026</footer>
</div></body></html>
"""

@site.route("/")
def index(): return render_template_string(HTML_PAGE, ai=AI_AVAILABLE)

@site.route("/generate", methods=["POST"])
def generate():
    topic=request.form.get("topic","").strip(); grade=int(request.form.get("grade",6))
    cols=int(request.form.get("cols",3)); rows=int(request.form.get("rows",6))
    title=request.form.get("title","").strip(); use_ai=request.form.get("use_ai")=="on"
    if not topic: flash("Введите тему!"); return redirect(url_for("index"))
    if not title: title=f"Лабиринт: {topic} ({grade} класс)"
    total=cols*rows-1
    if use_ai and AI_AVAILABLE: tasks=generate_tasks_ai(topic,grade,total)
    else: tasks=generate_tasks_fallback(topic,total)
    maze=MathMaze(cols,rows); maze.generate_path(); maze.assign_tasks(tasks)
    fname=f"output/maze_{uuid.uuid4().hex[:8]}.pdf"
    build_maze_pdf(maze,title,fname)
    return send_file(fname,as_attachment=True,download_name=f"labirint_{topic[:20]}.pdf",mimetype="application/pdf")

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    print(f"🌐 Сайт: http://127.0.0.1:{port}")
    site.run(host="0.0.0.0", port=port)
