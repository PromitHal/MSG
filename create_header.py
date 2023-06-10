class Header:
    def __init__(self,string:str):
        self.string=string
    def form_header(self):
            header = self.string
            for i in range(1, 21):
                header += f'mfcc_mean{i} '
                header+=  f'mfcc_var{i} '
            for j in range(1,129):
                header += f'mel_specgram_mean{j} '
                header += f'mel_specgram_var{j} '
            header +='  tempo'
            header += ' label'
            header = header.split()
            return header
    