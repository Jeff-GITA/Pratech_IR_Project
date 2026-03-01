import argparse
import os
import sys
from pathlib import Path
from pdf2image import convert_from_path
import torch
import torch.backends.cudnn as cudnn
import zipfile
import shutil
import gradio as gr



print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
# FILE = Path(__file__).resolve()
# FILE = Path(os.path.abspath('')).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = Path(os.path.abspath('')).resolve()  # YOLOv5 root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print("\nROOT:", ROOT)

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync



def yolo_detection(project_path, img_folder, doc_folder):
    #### Parameters ####
    # source = "../../Data/Example_Docs/img/doc/"
    # source = doc_folder #"../../Data/Example_Docs/p_img/doc/"
    source = os.path.join(project_path, img_folder, doc_folder)
    # output_path = ""
    project = os.path.join(project_path, "runs")
    # name = "detect"
    name = doc_folder
    exist_ok = True
    save_txt=False
    device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    weights = "weights/best.pt"
    dnn=False  # use OpenCV DNN for ONNX inference
    data = ROOT / 'data/coco128.yaml'
    half=False  # use FP16 half-precision inference
    imgsz=(640, 640)
    augment=False  # augmented inference
    conf_thres=0.3  # confidence threshold
    iou_thres=0.3  # NMS IOU threshold
    max_det=1000 
    classes=None
    agnostic_nms=False
    save_crop=False  # save cropped prediction boxes
    line_thickness=3  # bounding box thickness (pixels)
    save_conf=False  # save confidences in --save-txt labels
    nosave=False  # do not save images/videos
    view_img=False  # show results
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    update=False  # update all models
    visualize=False  # visualize features
    ###################################


    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    print("\nis_file, is_url, webcam:")
    print(is_file, is_url, webcam)

    #### Directories ####
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok, mkdir=True)  # increment run


    #### Load model ####
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size



    #### Dataloader ####
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs


    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    # return os.path.join(project_path, "runs")



def zip_folder(folder_path, output_path):
    
    try:
        # Create a ZipFile object
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk the directory tree
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    # Create the complete filepath of each file in the folder
                    file_path = os.path.join(root, file)
                    # Add file to zip archive
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))
        print(f"Folder '{folder_path}' successfully zipped into '{output_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

def read_files(files):
    return " - ".join(files)


def images_processing(doc_folder, dst_folder, img_dim=(640,640)):
    
    
    path_obj = Path(dst_folder)
    path_obj.mkdir(parents=True, exist_ok=True)

    ## List doc images ##
    img_list = os.listdir(doc_folder)
    # Sort the list of files by name
    img_list = sorted(img_list)

    ## Loop to iterate throught images ##
    for i in range(len(img_list)):
        img_file = os.path.join(doc_folder, img_list[i])
        img_save_file = os.path.join(dst_folder, img_list[i])
        
        img = img_processing(img_file, img_dim)        
        cv2.imwrite(img_save_file, img)


def img_processing(img_file, img_dim=(640,640)):
    
    ## Read image ##
    img = cv2.imread(img_file) # BGR
    ## Conver to gray scale ##
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert the grayscale image
    img = cv2.bitwise_not(img)
    ## Resize imgage ##
    img = cv2.resize(img, img_dim, interpolation=cv2.INTER_LANCZOS4)
    ## Conver image from gray scale to BGR (H,W,C) ##
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img    


def set_path(file_path_list):

    temp_project_path, file_name = os.path.split(file_path_list[0])
    return temp_project_path


def load_process_documents(file_path_list):
    
    print(f"\n\nDocuments: {file_path_list}\n\n")

    ## get pages ##
    pdf2img_folder = "pdf2img_folder"
    preprocessed_folder = "preprocessed_img"
    temp_project_path = set_path(file_path_list)

    for pdf_document in file_path_list:
        ###############
        #### Input ####
        ###############


        ##############
        #### Code ####
        ##############        
        # temp_project_path, file_name = os.path.split(pdf_document)
        pdf_doc_path, file_name = os.path.split(pdf_document)
        print(f"\n\nProject path: {temp_project_path}\n\n")
        dst_path = os.path.join(temp_project_path, pdf2img_folder)
        doc_name = file_name.split(".")[0]
        dst_folder = os.path.join(dst_path, doc_name)
        # print("pdf_document: ", pdf_document)
        # print("dst_folder: ", dst_folder)
        # print("doc_name: ", doc_name)
        
        pages = convert_from_path(pdf_path=pdf_document)#, poppler_path=poppler_path)
        

        #### actions ####
        # print("dst_folder:", dst_folder)
        path_obj = Path(dst_folder)
        # Check if the folder exists
        if path_obj.exists() and path_obj.is_dir():
            shutil.rmtree(path_obj)  # Removes the folder and all its contents
            print(f"Deleted folder: {path_obj}")
        else:
            print("Folder does not exist.")
        path_obj.mkdir(parents=True, exist_ok=True)
        p=1
        for page in pages:
            img_name = f"{doc_name}_{p}.png"
            # print("img_name: ",img_name)
            # print("folder_name:",os.path.join(dst_folder, img_name))
            p += 1
            page.save(os.path.join(dst_folder, img_name), "PNG")

        doc_folder = os.path.join(temp_project_path, pdf2img_folder, doc_name)
        preprocessed_img = os.path.join(temp_project_path, preprocessed_folder, doc_name)
    
        images_processing(doc_folder, preprocessed_img, img_dim=(640,640))
        yolo_detection(project_path=temp_project_path, img_folder=preprocessed_folder, doc_folder=doc_name)


    zip_file = os.path.join(temp_project_path, "output_detection.zip")
    print(f"\n\nTemporal folder:{temp_project_path} \n\n")
    print(zip_file)
    # Create a zip file (ZIP format by default)
    # shutil.make_archive(zip_name, 'zip', folder_to_zip)
    zip_folder(os.path.join(temp_project_path, "runs"), zip_file)
    
    return zip_file


iface = gr.Interface(
    fn=load_process_documents, 
    inputs=gr.File(label="Input_Files", file_count="multiple", type="filepath"), 
    outputs=gr.File(type="filepath"),
)
iface.launch(share=True)
# iface.launch(server_name="0.0.0.0")

