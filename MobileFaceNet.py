import torch
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import os
import cv2

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
   
    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)
       
        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        out = self.conv_6_sep(out)

        out = self.conv_6_dw(out)

        out = self.conv_6_flatten(out)

        out = self.linear(out)

        out = self.bn(out)
        return l2_norm(out)


# Example Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
try:
    model = MobileFaceNet(embedding_size=512).to(device)
    model.load_state_dict(torch.load("model_mobilefacenet.pth", map_location=device))
    model.eval()
except Exception as e:
    print(e)  # Set model to evaluation mode
# print(model.eval())  # Set model to evaluation mode





import numpy as np
from torchvision import transforms
from PIL import Image

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image)  # Get 512-D embedding
    return embedding.cpu().numpy()

def compare_faces(img1, emb2):
    emb1 = get_embedding(img1)
    # emb2 = get_embedding(img2)

    # Cosine Similarity (closer to 1 = similar)
    similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity


def recognize_face(embedding, dataset_embeddings):
    """Compare the given embedding with dataset embeddings."""
    best_match = None
    best_score = float("inf")  # Using Euclidean Distance (lower is better)

    for person, stored_embedding in dataset_embeddings.items():
        score = np.linalg.norm(embedding - stored_embedding)  # Euclidean Distance
        if score < best_score:
            best_match = person
            best_score = score

    return best_match, best_score

# Load dataset embeddings
dataset_path = "dataset"
# dataset_embeddings = load_dataset(dataset_path)
dataset_embeddings = np.load("embeddings.npy", allow_pickle=True).item()

# result = compare_faces("./dataset/Abbasi/face_0.jpg", "./dataset/Aayush/face_0.jpg")
# print("Cosine Similarity:", result)
        


from YuNetFace import FaceDetectorYunet

def start_webcam():
    rtsp_username = "admin"
    # rtsp_password = "123456789"
    rtsp_password = "cctv@123"
    width = 800
    height = 480
    cam_no = "1"
    rtsp = "rtsp://" + rtsp_username + ":" + rtsp_password + "@192.168.1.64:554/Streaming/channels/" + cam_no + "01"
    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    # cap.open(rtsp)
    cap.set(3, width)  # Set width
    cap.set(4, height)  # Set height
    # success, current_cam = cap.read()
    return cap


cap = cv2.VideoCapture(0)
# cap = start_webcam()
fd = FaceDetectorYunet()
students_marked = []
while True:

    ret, frame = cap.read()
    if not ret:
        break
        
    faces = fd.detect(frame)
    if faces:
        fd.draw_faces(frame, faces)

        for face in faces:
            try:
                x1, y1 = int(face["x1"]), int(face["y1"])
                x2, y2 = int(face["x2"]), int(face["y2"])
                w, h = x2 - x1, y2 - y1  # Compute width and height
            except KeyError:
                print(f"Skipping invalid face data: {face}")
                continue 
                
            face_roi = frame[y1:y1+h, x1:x1+w]
            if face_roi.size == 0:
                    continue
            cv2.imwrite("./face.jpg", face_roi)
            for person, stored_embedding in dataset_embeddings.items():
                result = compare_faces("./face.jpg", stored_embedding)
                if result > 0.7 and person not in students_marked:  
                    print(f"Person: {person}, Score: {result}")
                    students_marked.append(person)
            # result = compare_faces("./face.jpg", "./dataset/Aayush/face_0.jpg")
            # print("Cosine Similarity:", result)

        # Compare two images
        # for img in faces:
        #     result = compare_faces(img, "./dataset/Aayush/face_0.jpg")
        #     print("Cosine Similarity:", result)
        # print(type(faces))


    cv2.imshow('YuNet Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



