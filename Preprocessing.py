import os
import numpy as np

def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

def normalize_blinks(num_blinks, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur, u_Dur, sigma_Dur, Vel, u_Vel,
                     sigma_Vel,Five,u_five,sigma_five,six,u_six,sigma_six,seven,u_seven,sigma_seven,eight,u_eight,sigma_eight):
    # input is the blinking features as well as their mean and std, mfcc4 is a [num_blinksx4] matrix as the normalized blinks
    normalized_blinks = np.zeros([num_blinks, 8])
    normalized_Freq = (Freq[0:num_blinks] - u_Freq) / sigma_Freq
    normalized_blinks[:, 0] = normalized_Freq
    normalized_Amp = (Amp[0:num_blinks]  - u_Amp) / sigma_Amp
    normalized_blinks[:, 1] = normalized_Amp
    normalized_Dur = (Dur[0:num_blinks]  - u_Dur) / sigma_Dur
    normalized_blinks[:, 2] = normalized_Dur
    normalized_Vel = (Vel[0:num_blinks]  - u_Vel) / sigma_Vel
    normalized_blinks[:, 3] = normalized_Vel
    normalized_five = (Five[0:num_blinks] - u_five) / sigma_five
    normalized_blinks[:, 4] = normalized_five
    normalized_six = (six[0:num_blinks] - u_six) / sigma_six
    normalized_blinks[:, 5] = normalized_six
    normalized_seven = (seven[0:num_blinks] - u_seven) / sigma_seven
    normalized_blinks[:, 6] = normalized_seven
    normalized_eight = (eight[0:num_blinks] - u_eight) / sigma_eight
    normalized_blinks[:, 7] = normalized_eight

    return normalized_blinks


def unroll_in_time(in_data, window_size, stride):
    # in_data is [n,4]            out_data is [N,Window_size,4]
    n = len(in_data)
    if n <= window_size:
        out_data = np.zeros([1, window_size,8])
        out_data[0, -n:, :] = in_data
        return out_data
    else:
        N = ((n - window_size) // stride) + 1
        out_data = np.zeros([N, window_size, 8])
        for i in range(N):
            if i * stride + window_size <= n:
                out_data[i, :, :] = in_data[i * stride:i * stride + window_size, :]
            else:  # this line should not ever be executed because of the for mula used above N is the exact time the loop is executed
                break

        return out_data

def gen(folder_list,window_size,stride,path1):
    for ID, folder in enumerate(folder_list):
        print("#########\n")
        print(str(ID) + '-' + str(folder) + '\n')
        print("#########\n")
        files_per_person = os.listdir(path1 + '/' + folder)
        files_per_person.sort()
        for txt_file in files_per_person:
            if txt_file == 'xiaobo_gaze_2_w50_o30_alert.txt':
                alertTXT = path1 + '/' + folder + '/' + txt_file
                Freq = np.loadtxt(alertTXT, usecols=0)
                Amp = np.loadtxt(alertTXT, usecols=1)
                Dur = np.loadtxt(alertTXT, usecols=2)
                Vel = np.loadtxt(alertTXT, usecols=3)
                five=np.loadtxt(alertTXT,usecols=4)
                six=np.loadtxt(alertTXT,usecols=5)
                seven = np.loadtxt(alertTXT, usecols=6)
                eight = np.loadtxt(alertTXT, usecols=7)
                blink_num = len(Freq)
                bunch_size = blink_num // 3  # one third used for baselining
                remained_size = blink_num - bunch_size
                # Using the last bunch_size number of blinks to calculate mean and std
                u_Freq = np.mean(Freq[-bunch_size:])
                sigma_Freq = np.std(Freq[-bunch_size:])
                if sigma_Freq == 0:
                    sigma_Freq = np.std(Freq)
                u_Amp = np.mean(Amp[-bunch_size:])
                sigma_Amp = np.std(Amp[-bunch_size:])
                if sigma_Amp == 0:
                    sigma_Amp = np.std(Amp)
                u_Dur = np.mean(Dur[-bunch_size:])

                sigma_Dur = np.std(Dur[-bunch_size:])
                if sigma_Dur == 0:
                    sigma_Dur = np.std(Dur)
                u_Vel = np.mean(Vel[-bunch_size:])
                sigma_Vel = np.std(Vel[-bunch_size:])
                if sigma_Vel == 0:
                    sigma_Vel = np.std(Vel)
                u_five = np.mean(five[-bunch_size:])
                sigma_five = np.std(five[-bunch_size:])
                if sigma_five== 0:
                    sigma_five = np.std(five)
                u_six = np.mean(six[-bunch_size:])
                sigma_six = np.std(six[-bunch_size:])
                if sigma_six == 0:
                    sigma_six = np.std(six)
                u_seven = np.mean(seven[-bunch_size:])
                sigma_seven = np.std(seven[-bunch_size:])
                if sigma_seven == 0:
                    sigma_seven = np.std(seven)
                u_eight = np.mean(eight[-bunch_size:])
                sigma_eight = np.std(eight[-bunch_size:])
                if sigma_eight == 0:
                    sigma_eight = np.std(eight)
                print('freq: %f, amp: %f, dur: %f, vel: %f ,vel: %f ,vel: %f vel: %f ,vel: %f \n' % (u_Freq, u_Amp, u_Dur, u_Vel,u_five,u_six,u_seven,u_eight))
                normalized_blinks = normalize_blinks(remained_size, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp,
                                                     Dur, u_Dur, sigma_Dur,
                                                              Vel, u_Vel, sigma_Vel,five,u_five,sigma_five,six,u_six,sigma_six,seven,u_seven,sigma_seven,eight,u_eight,sigma_eight)

                print('Postfreq: %f, Postamp: %f, Postdur: %f, Postvel: %f ,Postvel: %f Postvel: %f Postvel: %f Postvel: %f \n' % (np.mean(normalized_blinks[:, 0]),
                                                                                  np.mean(normalized_blinks[:, 1]),
                                                                                  np.mean(normalized_blinks[:, 2]),
                                                                                  np.mean(normalized_blinks[:, 3]),np.mean(normalized_blinks[:, 4]),np.mean(normalized_blinks[:, 5]),np.mean(normalized_blinks[:, 6]),np.mean(normalized_blinks[:, 7])))

                alert_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                # sweep a window over the blinks to chunk
                alert_labels = 0 * np.ones([len(alert_blink_unrolled), 1])

            if txt_file == 'xiaobo_gaze_2_w50_o30_semisleepy.txt':
                blinksTXT = path1 + '/' + folder + '/' + txt_file
                Freq = np.loadtxt(blinksTXT, usecols=0)
                Amp = np.loadtxt(blinksTXT, usecols=1)
                Dur = np.loadtxt(blinksTXT, usecols=2)
                Vel = np.loadtxt(blinksTXT, usecols=3)
                five=np.loadtxt(blinksTXT,usecols=4)
                six=np.loadtxt(blinksTXT,usecols=5)
                seven = np.loadtxt(blinksTXT, usecols=6)
                eight = np.loadtxt(blinksTXT, usecols=7)
                blink_num = len(Freq)

                normalized_blinks = normalize_blinks(blink_num, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur,
                                                     u_Dur, sigma_Dur, Vel, u_Vel, sigma_Vel,five,u_five,sigma_five,six,u_six,sigma_six,seven,u_seven,sigma_seven,eight,u_eight,sigma_eight)
                print('SEMIfreq: %f, SEMIamp: %f, SEMIdur: %f, SEMIvel: %f SEMIvel: %f SEMIvel: %f SEMIvel: %f SEMIvel: %f\n' % (np.mean(normalized_blinks[:, 0]),
                                                                                  np.mean(normalized_blinks[:, 1]),
                                                                                  np.mean(normalized_blinks[:, 2]),
                                                                                  np.mean(normalized_blinks[:, 3]),np.mean(normalized_blinks[:, 4]),np.mean(normalized_blinks[:, 5]),np.mean(normalized_blinks[:, 6]),np.mean(normalized_blinks[:, 7])))

                semi_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                semi_labels = 5 * np.ones([len(semi_blink_unrolled), 1])

            if txt_file == 'xiaobo_gaze_2_w50_o30_sleepy.txt':
                blinksTXT = path1 + '/' + folder + '/' + txt_file
                Freq = np.loadtxt(blinksTXT, usecols=0)
                Amp = np.loadtxt(blinksTXT, usecols=1)
                Dur = np.loadtxt(blinksTXT, usecols=2)
                Vel = np.loadtxt(blinksTXT, usecols=3)
                five=np.loadtxt(blinksTXT,usecols=4)
                six=np.loadtxt(blinksTXT,usecols=5)
                seven = np.loadtxt(blinksTXT, usecols=6)
                eight = np.loadtxt(blinksTXT, usecols=7)
                blink_num = len(Freq)

                normalized_blinks = normalize_blinks(blink_num, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur,
                                                     u_Dur, sigma_Dur, Vel, u_Vel, sigma_Vel,five,u_five,sigma_five,six,u_six,sigma_six,seven,u_seven,sigma_seven,eight,u_eight,sigma_eight)
                print(
                'SLEEPYfreq: %f, SLEEPYamp: %f, SLEEPYdur: %f, SLEEPYvel: %f SLEEPYvel: %f SLEEPYvel: %f  SLEEPYvel: %f SLEEPYvel: %f\n' % (np.mean(normalized_blinks[:, 0]),
                                                                                    np.mean(normalized_blinks[:, 1]),
                                                                                    np.mean(normalized_blinks[:, 2]),
                                                                                    np.mean(normalized_blinks[:, 3]),np.mean(normalized_blinks[:, 4]),np.mean(normalized_blinks[:, 5]),np.mean(normalized_blinks[:, 6]),np.mean(normalized_blinks[:, 7])))

                sleepy_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                sleepy_labels = 10 * np.ones([len(sleepy_blink_unrolled), 1])

        tempX = np.concatenate((alert_blink_unrolled, semi_blink_unrolled, sleepy_blink_unrolled), axis=0)
        tempY = np.concatenate((alert_labels, semi_labels, sleepy_labels), axis=0)
        if ID > 0:
            output = np.concatenate((output, tempX), axis=0)
            labels = np.concatenate((labels, tempY), axis=0)
        else:
            output = tempX
            labels = tempY
    return output,labels



def Preprocess(path1,window_size,stride,test_fold):
    #path1 is the address to the folder of all subjects, each subject has three txt files for alert, semisleepy and sleepy levels
    #window_size decides the length of blink sequence
    #stride is the step by which the moving windo slides over consecutive blinks to generate the sequences
    #test_fold is the number of fold that is picked as test and uses the other folds as training
    #mfcc4=[N,T,F]
    path=path1
    folds_list = os.listdir(path1)
    folds_list.sort()
    for f, fold in enumerate(folds_list):
        print(fold)
        path1 = path + '/' + fold
        folder_list = os.listdir(path1)
        folder_list.sort()
        if fold==test_fold:
            outTest,labelTest=gen(folder_list,window_size,stride,path1)
            print("Not this fold ;)")
            continue
        for ID,folder in enumerate(folder_list):
            print("#########\n")
            print(str(ID)+'-'+ str(folder)+'\n')
            print("#########\n")
            files_per_person = os.listdir(path1 + '/' + folder)
            files_per_person.sort()
            for txt_file in files_per_person:
                if txt_file=='xiaobo_gaze_2_w50_o30_alert.txt':
                    alertTXT = path1 + '/' + folder + '/' + txt_file
                    Freq = np.loadtxt(alertTXT, usecols=0)
                    Amp = np.loadtxt(alertTXT, usecols=1)
                    Dur = np.loadtxt(alertTXT, usecols=2)
                    Vel = np.loadtxt(alertTXT, usecols=3)
                    Five = np.loadtxt(alertTXT,usecols=4)
                    six = np.loadtxt(alertTXT, usecols=5)
                    seven = np.loadtxt(alertTXT, usecols=6)
                    eight = np.loadtxt(alertTXT, usecols=7)
                    blink_num=len(Freq)
                    bunch_size=blink_num // 3   #one third used for baselining
                    remained_size=blink_num-bunch_size
                    # Using the last bunch_size number of blinks to calculate mean and std
                    u_Freq=np.mean(Freq[-bunch_size:])
                    sigma_Freq=np.std(Freq[-bunch_size:])
                    if sigma_Freq==0:
                        sigma_Freq=np.std(Freq)
                    u_Amp=np.mean(Amp[-bunch_size:])
                    sigma_Amp=np.std(Amp[-bunch_size:])
                    if sigma_Amp==0:
                        sigma_Amp=np.std(Amp)
                    u_Dur=np.mean(Dur[-bunch_size:])

                    sigma_Dur=np.std(Dur[-bunch_size:])
                    if sigma_Dur==0:
                        sigma_Dur=np.std(Dur)
                    u_Vel=np.mean(Vel[-bunch_size:])
                    sigma_Vel=np.std(Vel[-bunch_size:])
                    if sigma_Vel==0:
                        sigma_Vel=np.std(Vel)
                    u_five = np.mean(Five[-bunch_size:])
                    sigma_five = np.std(Five[-bunch_size:])
                    if sigma_five == 0:
                        sigma_five = np.std(Five)
                    u_six = np.mean(six[-bunch_size:])
                    sigma_six = np.std(six[-bunch_size:])
                    if sigma_six == 0:
                        sigma_six = np.std(six)
                    u_seven = np.mean(seven[-bunch_size:])
                    sigma_seven = np.std(seven[-bunch_size:])
                    if sigma_seven == 0:
                        sigma_seven = np.std(seven)
                    u_eight = np.mean(eight[-bunch_size:])
                    sigma_eight = np.std(eight[-bunch_size:])
                    if sigma_eight == 0:
                        sigma_eight = np.std(eight)
                    print('freq: %f, amp: %f, dur: %f, vel: %f,five:%f,six:%f  five:%f,six:%f  \n' %(u_Freq,u_Amp,u_Dur,u_Vel,u_five,u_six,u_seven,u_eight))
                    normalized_blinks=normalize_blinks(remained_size, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur, u_Dur, sigma_Dur,
                                     Vel, u_Vel, sigma_Vel,Five,u_five,sigma_five,six,u_six,sigma_six,seven,u_seven,sigma_seven,eight,u_eight,sigma_eight)

                    print('Postfreq: %f, Postamp: %f, Postdur: %f, Postvel: %f ,postfive:%f,postsix:%f postfive:%f,postsix:%f\n' % (np.mean(normalized_blinks[:,0]),
                                                                                      np.mean(normalized_blinks[:,1]),
                                                                                      np.mean(normalized_blinks[:,2]),
                                                                                      np.mean(normalized_blinks[:,3]),np.mean(normalized_blinks[:,4]),np.mean(normalized_blinks[:,5]),np.mean(normalized_blinks[:,6]),np.mean(normalized_blinks[:,7])))

                    alert_blink_unrolled=unroll_in_time(normalized_blinks,window_size,stride)
                    # sweep a window over the blinks to chunk
                    alert_labels = 0 * np.ones([len(alert_blink_unrolled), 1])




                if txt_file=='xiaobo_gaze_2_w50_o30_semisleepy.txt':
                    blinksTXT = path1 + '/' + folder + '/' + txt_file
                    Freq = np.loadtxt(blinksTXT, usecols=0)
                    Amp = np.loadtxt(blinksTXT, usecols=1)
                    Dur = np.loadtxt(blinksTXT, usecols=2)
                    Vel = np.loadtxt(blinksTXT, usecols=3)
                    five=np.loadtxt(blinksTXT,usecols=4)
                    six=np.loadtxt(blinksTXT,usecols=5)
                    seven = np.loadtxt(blinksTXT, usecols=6)
                    eight = np.loadtxt(blinksTXT, usecols=7)
                    blink_num = len(Freq)


                    normalized_blinks = normalize_blinks(blink_num, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur,
                                                     u_Dur, sigma_Dur,Vel, u_Vel, sigma_Vel,five,u_five,sigma_five,six,u_six,sigma_six,seven,u_seven,sigma_seven,eight,u_eight,sigma_eight)
                    print('SEMIfreq: %f, SEMIamp: %f, SEMIdur: %f, SEMIvel: %f ,semifive:%f,semisix:%f semifive:%f,semisix:%f\n' % (np.mean(normalized_blinks[:,0]),
                                                                                      np.mean(normalized_blinks[:,1]),
                                                                                      np.mean(normalized_blinks[:,2]),
                                                                                      np.mean(normalized_blinks[:,3]),np.mean(normalized_blinks[:,4]),np.mean(normalized_blinks[:,5]),np.mean(normalized_blinks[:,6]),np.mean(normalized_blinks[:,7])))

                    semi_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                    semi_labels = 5* np.ones([len(semi_blink_unrolled), 1])

                if txt_file == 'xiaobo_gaze_2_w50_o30_sleepy.txt':
                    blinksTXT = path1 + '/' + folder + '/' + txt_file
                    Freq = np.loadtxt(blinksTXT, usecols=0)
                    Amp = np.loadtxt(blinksTXT, usecols=1)
                    Dur = np.loadtxt(blinksTXT, usecols=2)
                    Vel = np.loadtxt(blinksTXT, usecols=3)
                    five=np.loadtxt(blinksTXT,usecols=4)
                    six=np.loadtxt(blinksTXT,usecols=5)
                    seven = np.loadtxt(blinksTXT, usecols=6)
                    eight = np.loadtxt(blinksTXT, usecols=7)
                    blink_num = len(Freq)

                    normalized_blinks = normalize_blinks(blink_num, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp,
                                                         Dur,
                                                         u_Dur, sigma_Dur, Vel, u_Vel, sigma_Vel, five, u_five,
                                                         sigma_five, six, u_six, sigma_six, seven, u_seven, sigma_seven,
                                                         eight, u_eight, sigma_eight)
                    print(
                        'SEMIfreq: %f, SEMIamp: %f, SEMIdur: %f, SEMIvel: %f ,semifive:%f,semisix:%f semifive:%f,semisix:%f\n' % (
                        np.mean(normalized_blinks[:, 0]),
                        np.mean(normalized_blinks[:, 1]),
                        np.mean(normalized_blinks[:, 2]),
                        np.mean(normalized_blinks[:, 3]), np.mean(normalized_blinks[:, 4]),
                        np.mean(normalized_blinks[:, 5]), np.mean(normalized_blinks[:, 6]),
                        np.mean(normalized_blinks[:, 7])))

                sleepy_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                sleepy_labels=10*np.ones([len(sleepy_blink_unrolled),1])


            tempX=np.concatenate((alert_blink_unrolled,semi_blink_unrolled,sleepy_blink_unrolled),axis=0)
            tempY = np.concatenate((alert_labels, semi_labels, sleepy_labels), axis=0)
            if test_fold!="Fold1":
                start=0
            else:
                start=1
            if f !=start  or ID>0:
                output=np.concatenate((output,tempX),axis=0)
                labels=np.concatenate((labels,tempY),axis=0)
            else:
                output=tempX
                labels=tempY

    output,labels=unison_shuffled_copies(output,labels)
    print('We have %d training datapoints!!!' %len(labels))
    print('We have %d test datapoints!!!' % len(labelTest))
    print('We have in TOTAL %d datapoints!!!' % (len(labelTest)+len(labels)))
    return output,labels,outTest,labelTest

#path1 is the address to the folder of all subjects, each subject has three txt files for alert, semisleepy and sleepy levels
path1='xbgazew50o30wei2'
window_size=30
stride=2
for i in range(1,6):
    Training='xbgazew50o30wei2npy/xiaobo_30_Fold%d.npy'%i
    Testing='xbgazew50o30wei2npy/xiaoboTest_30_Fold%d.npy'%i
    label='xbgazew50o30wei2npy/Labels_30_Fold%d.npy'%i
    labeltest='xbgazew50o30wei2npy/LabelsTest_30_Fold%d.npy'%i
#################Normalizing with respect to different individuals####First Phase
    test_fold='Fold%d'%i
    xiaobo,labels,xiaoboTest,labelTest=Preprocess(path1,window_size,stride,test_fold=test_fold)
    np.save(open(Training,'wb'),xiaobo)
    np.save(open(label, 'wb'),labels)
    np.save(open(Testing, 'wb'),xiaoboTest)
    np.save(open(labeltest, 'wb'),labelTest)

