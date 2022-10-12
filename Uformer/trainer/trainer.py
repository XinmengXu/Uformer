import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
from tqdm import tqdm
from trainer.base_trainer import BaseTrainer
from model.DCCRN.ConvSTFT import ConvSTFT, ConviSTFT 
#from util.utils import compute_STOI, compute_PESQ
plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader
        self.stft = ConvSTFT(512, 256, 512, 'hamming', 'complex').to(self.device)
        self.istft = ConviSTFT(512, 256, 512, 'hamming', 'complex').to(self.device)
    def _train_epoch(self, epoch):
        loss_total = 0.0
        num_batchs = len(self.train_data_loader)
        start_time = time.time()
        with tqdm(total = num_batchs) as pbar:
            for i, (mixture, clean, name) in enumerate(self.train_data_loader):
                mixture = mixture.to(self.device)
                clean = clean.to(self.device)

                self.optimizer.zero_grad()
                enhanced_cpl, enhanced = self.model(mixture)
                loss = self.model.loss(enhanced, clean, mode='SiSNR')
                #loss, _, _ = self.model.loss(enhanced_cpl, clean, mode='Mix')
                #print(loss)
                #print(amp_loss)
                #print(phase_loss)
                loss.backward()
                self.optimizer.step()

                loss_total += loss.item()
                pbar.update(1)
            end_time = time.time()
            dl_len = len(self.train_data_loader)
            print("loss:", loss_total / dl_len)         
            self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)
 
    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]
        num_batchs = len(self.train_data_loader)
        sample_length = self.validation_custom_config["sample_length"]
        self.stft = ConvSTFT(512, 256, 512, 'hamming', 'complex').to(self.device)
        self.istft = ConviSTFT(400, 100, 512, 'hamming', 'complex').to(self.device)
        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []
        with tqdm(total = num_batchs) as pbar:  
            for i, (mixture, clean, name) in enumerate(self.validation_data_loader):
                assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
                name = name[0]
                padded_length = 0

                mixture = mixture.to(self.device)  # [1, 1, T]
                clean = clean.to(self.device)                
                enhanced_cpl, enhanced = self.model(mixture)
                #print(enhanced_cpl)
                loss = self.model.loss(enhanced, clean, mode='SiSNR')
                #loss, _, _ = self.model.loss(enhanced_cpl, clean, mode='Mix')
                loss_total += loss.item()

                #enhanced = self.istft(enhanced_cpl, None)


                enhanced = enhanced.reshape(-1).cpu().numpy()
                clean = clean.cpu().numpy().reshape(-1)
                mixture = mixture.cpu().numpy().reshape(-1)
                assert len(mixture) == len(enhanced) == len(clean)


            # Visualize audio
                if i <= visualize_audio_limit:
                    self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=16000)
                    self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=16000)
                    self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=16000)

                # Visualize waveform
                if i <= visualize_waveform_limit:
                    fig, ax = plt.subplots(3, 1)
                    for j, y in enumerate([mixture, enhanced, clean]):
                        ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                            np.mean(y),
                            np.std(y),
                            np.max(y),
                            np.min(y)
                        ))
                        librosa.display.waveplot(y, sr=16000, ax=ax[j])
                    plt.tight_layout()
                    self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram
                noisy_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160, win_length=320))
                enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320))
                clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))

                if i <= visualize_spectrogram_limit:
                    fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                    for k, mag in enumerate([
                        noisy_mag,
                        enhanced_mag,
                        clean_mag,
                    ]):
                        axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                          f"std: {np.std(mag):.3f}, "
                                          f"max: {np.max(mag):.3f}, "
                                          f"min: {np.min(mag):.3f}")
                        librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=16000)
                    plt.tight_layout()
                    self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)
                output_path = os.path.join('H:/data3/r3', f"{name}.wav")
                librosa.output.write_wav(output_path, enhanced, sr = 16000)
                pbar.update(1)
            # Metric
            # stoi_c_n.append(compute_STOI(clean, mixture, sr=16000))
            # stoi_c_e.append(compute_STOI(clean, enhanced, sr=16000))
            # pesq_c_n.append(compute_PESQ(clean, mixture, sr=16000))
            # pesq_c_e.append(compute_PESQ(clean, enhanced, sr=16000))

        # get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        # self.writer.add_scalars(f"Metric/STOI", {
            # "Clean and noisy": get_metrics_ave(stoi_c_n),
            # "Clean and enhanced": get_metrics_ave(stoi_c_e)
        # }, epoch)
        # self.writer.add_scalars(f"Metric/PESQ", {
            # "Clean and noisy": get_metrics_ave(pesq_c_n),
            # "Clean and enhanced": get_metrics_ave(pesq_c_e)
        # }, epoch)

        score = loss_total
        return score
