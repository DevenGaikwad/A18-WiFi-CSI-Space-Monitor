"""Exploratory feature-removal audit; preserves the recorded split and parameters."""
import argparse,csv,io,json,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.model_selection import LeaveOneGroupOut

LABELS=['NO_PRESENCE','PERSON_PRESENT','PERSON_MOVING']
def load(archive_path):
    X=[];y=[];groups=[]
    with zipfile.ZipFile(archive_path) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC failed')
        names=sorted(n for n in z.namelist() if n.lower().endswith('.csv'))
        if len(names)!=45:raise ValueError('Expected 45 sessions')
        for j,n in enumerate(names):
            rows=list(csv.DictReader(io.StringIO(z.read(n).decode('utf-8-sig'))))
            label=rows[0]['label'];sid=rows[0]['session_id'];windows=[[] for _ in range(28)]
            for r in rows:
                if not (r['label']==label and r['session_id']==sid and r['tx_mac'].lower()=='a4:f0:0f:69:1a:94' and r['channel']=='6' and r['csi_len']=='128' and r['first_word_invalid']=='1' and len(r['csi_hex'])==256):raise ValueError(n)
                w=(int(r['rx_timestamp_us'])-20_000_000)//2_000_000
                if 0<=w<28:
                    b=np.frombuffer(bytes.fromhex(r['csi_hex']),dtype=np.int8).astype(float)
                    windows[w].append(np.hypot(b[4::2],b[5::2]))
            for frames in windows:
                if len(frames)<10:raise ValueError('Too few frames')
                a=np.asarray(frames);flat=a.ravel();change=np.abs(np.diff(a,axis=0))
                X.append([flat.mean(),flat.std(),np.percentile(flat,10),np.percentile(flat,90),a.mean(axis=0).std(),a.std(axis=0).mean(),change.mean(),np.percentile(change,90)])
                y.append(label);groups.append(j//3+1)
    X=np.asarray(X);y=np.asarray(y);groups=np.asarray(groups)
    for g in range(1,16):
        if Counter(y[groups==g])!=Counter({l:28 for l in LABELS}):raise ValueError('Incomplete round')
    return X,y,groups
def rf():return RandomForestClassifier(n_estimators=300,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=-1)
def score(y,p):return dict(accuracy=accuracy_score(y,p),macro_f1=f1_score(y,p,labels=LABELS,average='macro'),confusion=confusion_matrix(y,p,labels=LABELS).tolist())
def run(path,out):
    X,y,g=load(path);train=g<=13;test=g>=14;vp=np.empty(train.sum(),dtype=object)
    for a,b in LeaveOneGroupOut().split(X[train],y[train],g[train]):vp[b]=rf().fit(X[train][a],y[train][a]).predict(X[train][b])
    model=rf().fit(X[train],y[train]);tp=model.predict(X[test])
    result=dict(description='Exploratory CSI amplitude and temporal features only; RSSI mean and std removed; same fixed splits and RF parameters; baseline model retained',validation=score(y[train],vp),held_out=score(y[test],tp))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('--out',type=Path,default=Path('results/exploratory/csi_only_ablation_reproduced.json'))
    a=parser.parse_args();print(json.dumps(run(a.archive,a.out),indent=2))
