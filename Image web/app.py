from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import hashlib
import numpy as np

def text_to_md5(text):
    md5_hash=hashlib.md5()
    md5_hash.update(text.encode('utf-8'))
    return md5_hash.hexdigest()

def generate_random_matrix(m, n, seed):
    np.random.seed(seed)
    return np.random.rand(m, n)

def encrypt(ssmsg,image,key, key_matrix):
    
    msg= ssmsg + key
    secretebits =''
    for i in msg:
        x=ord(i)
        b=bin(x)[2:]
        secretebits+=b.zfill(8)

    image2=image.copy()
    m,n,_=image.shape

    counter=0
    for i in range(m):
        for j in range(n):
            for k in range(3):
                if counter==len(secretebits):
                    break
                pixel=image[i,j,k]
                kv = key_matrix[i,j]
                b=bin(pixel)[2:].zfill(8)
                bk=bin(kv)[2:].zfill(8)
                bs = secretebits[counter]
                xorv=int(bk[-1]) ^ int(bs)
                r=b[0:-1] + str(xorv)
                d=int(r,2)
                image2[i,j,k]=d
                counter+=1
    return image2

def decrypt_msg(image, key, key_matrix):
    y,z,_= image.shape
    retrieved=''
    counter=0
    for p in range(y):
        for q in range(z):
            for r in range(3):
                if counter>=image.shape[0]*image.shape[1]*3*8:
                    break
                pixel=image[p,q,r]
                kv=key_matrix[p,q]
                bk=bin(kv)[2:].zfill(8)
                b2=bin(pixel)[2:].zfill(8)
                lsb=int(b2[-1])
                retrieved+=str(lsb ^ int(bk[-1]))
                counter+=1

    msgg=''
    for i in range(0,len(retrieved),8):
        x=retrieved[i:i+8]
        c=chr(int(x,2))
        
        msgg+=c
        l=len(msgg)

    counter=0
    
    for tt in msgg:
        if counter>len(key):
            if msgg[counter-len(key):counter]==key:
                break
        counter+=1
                
    return msgg[:counter-len(key)]

def generate_random_matrix(m, n, seed):
    np.random.seed(seed)
    return np.random.rand(m, n)

def encrypt2(img,image,key, key_matrix,s2,s1):
    y,z,_=img.shape
    secrete=''
    for i in range(y):
        for j in range(z):
            for k in range(3):
                pixel=img[i,j,k]
                b=bin(pixel)[2:].zfill(8)
                secrete+=b

    for k in key:
        
        x=ord(k)
        b=bin(x)[2:]
        secrete+=b.zfill(8)
    
    b=bin(s2)[2:].zfill(8)
    secrete+=b

    b=bin(s1)[2:].zfill(8)
    secrete+=b
    
    m,n,_=image.shape
    copiedimage=image.copy()
  
    counter=0
    for p in range(m):
        for q in range(n):
            for r in range(3):
                if counter==len(secrete):
                    break
                pixell=image[p,q,r]
                kv = key_matrix[p,q]
                b2=bin(pixell)[2:].zfill(8)
                bk=bin(kv)[2:].zfill(8)
                bs = secrete[counter]
                xorv=int(bk[-1]) ^ int(bs)
                rr=b2[0:-1] + str(xorv)
                d=int(rr,2)
                
                copiedimage[p,q,r]=d
                counter+=1  
    return copiedimage

def generate_random_matrix(a, b, seed):
    np.random.seed(seed)
    return np.random.rand(a, b)

def decryptimg(image, key, key_matrix):
    a,b,_= image.shape
    retrieved=''
    counter=0
    secrete=''
    for k in key:
        
        x=ord(k)
        bb=bin(x)[2:]
        secrete+=bb.zfill(8)

    print(secrete)
    for e in range(a):
        for f in range(b):
            for g in range(3):

                
                pixel=image[e,f,g]
                kv=key_matrix[e,f]
                bk=bin(kv)[2:].zfill(8)
                b2=bin(pixel)[2:].zfill(8)
                lsb=int(b2[-1])
                retrieved+=str(lsb ^ int(bk[-1]))

                if len(retrieved) > 8*len(key):
                    
                    xx=retrieved[len(retrieved)-8*len(key):]
                    if xx==secrete:   
                        scounter=counter
                counter+=1
    
    #s2=int(retrieved[scounter:scounter+8],2)
    #s1=int(retrieved[scounter+8:scounter+16],2)

    s1=128
    s2=128

    print(s1,s2)
    #scounter=int(s1*s2*3)
    
    msgimage=[]
    for i in range(0,scounter,8):
        g=retrieved[i:i+8]
        ch=int(g,2)
        msgimage.append(ch)
    l=len(msgimage)
    counter=0
    newimage=np.zeros((s1, s2, 3),dtype=np.uint8)
    for u in range(newimage.shape[0]):
        for v in range(newimage.shape[1]):
            for w in range(3):
                if counter < len(msgimage):
                    newimage[u,v,w]=msgimage[counter]
                    counter+=1
    return newimage  

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=["POST","GET"])
def home():
    if request.method == 'POST':
        message = request.form['message']
        file = request.files['file']
        key = request.form['key']

        hash_value = text_to_md5(key)

        seed = int(hash_value[:8], 16)

        #if file and file.filename.endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        image = cv2.imread(filepath)

        m,n,_=image.shape

        key_matrix = (255*generate_random_matrix(m, n, seed)).astype(int)


        encrypt_image= encrypt(message,image,key,key_matrix)
        saved_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
        cv2.imwrite(saved_filepath, encrypt_image)
        
        processed_image_path = 'processed_' + file.filename
        org_path = file.filename
        print('p:  ',processed_image_path)
        return render_template('index.html', processed_image=processed_image_path,imsg=message,ikey=key,org_image=org_path)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def generate_random_matrix(y, z, seed):
    np.random.seed(seed)
    return np.random.rand(y, z)


    
@app.route('/decrypt', methods=["POST","GET"])
def decrypt():
    if request.method == 'POST':
        file = request.files['decryptfile']
        key = request.form['decryptkey']

        hash_value = text_to_md5(key)
        seed = int(hash_value[:8], 16)

        #if file and file.filename.endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        image = cv2.imread(filepath)

        
        y,z,_= image.shape
        key_matrix = (255*generate_random_matrix(y, z, seed)).astype(int)
        
        msg=decrypt_msg(image, key, key_matrix)

        dec_path = file.filename
        return render_template('index.html',msg=msg,dkey=key,dec_image=dec_path)





@app.route('/image2image', methods=["POST","GET"])
def image2image():
    if request.method == 'POST':
        file1 = request.files['file1']
        key = request.form['key']
        hash_value = text_to_md5(key)
        seed = int(hash_value[:8], 16)
        #if file and file.filename.endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(filepath)
        img = cv2.imread(filepath)

        w,h,_ = img.shape
        s1,s2 = w//3,h//3
        s1,s2 = 128,128
        
        img=cv2.resize(img, (s1, s2))

        file2 = request.files['file2']
        #if file and file.filename.endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
        file2.save(filepath)
        image = cv2.imread(filepath)

        m,n,_=image.shape
        key_matrix = (255*generate_random_matrix(m, n, seed)).astype(int)

        encrypt_image= encrypt2(img,image,key,key_matrix,s2,s1)
        saved_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file2.filename)
        cv2.imwrite(saved_filepath, encrypt_image)

        processed_image_path = 'processed_' + file2.filename
        org_path1 = file1.filename
        org_path2 = file2.filename

        return render_template('image.html', processed_image=processed_image_path, ikey=key, org_image1=org_path1, org_image2=org_path2)
    return render_template('image.html')


         
    
@app.route('/decrypt_img', methods=["POST","GET"])
def decrypt_img():
    if request.method == 'POST':
        file = request.files['decryptfile']
        key = request.form['decryptkey']

        hash_value = text_to_md5(key)
        seed = int(hash_value[:8], 16)

        #if file and file.filename.endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        image = cv2.imread(filepath)

        
        a,b,_= image.shape
        key_matrix = (255*generate_random_matrix(a,b, seed)).astype(int)
        
        img = decryptimg(image, key, key_matrix)

        saved_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'decrypt_' + file.filename)
        cv2.imwrite(saved_filepath, img)
        processed_image_path = 'decrypt_' + file.filename

        #return send_file(img, mimetype='image/png')
        return render_template('image.html',dkey=key,decrypted_image_url=processed_image_path)
    
if __name__ == '__main__':
    app.run(debug=True)
