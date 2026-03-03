import requests

def unstreamfunc(modelname,pdf_text):
    document_content = pdf_text
    prompt = """
We have found that the reaction of 4,4'- dimethyltriphenylamine with methacryloyl chloride under Friedel-Crafts acylation condition to give BBMP by chance. Plausible mechanism to produce BBMP was shown in Scheme 1. After the production of the acylated compound, dimerization was thought to take place to give BBMP under acidic condition. Unfortunately, the reproducibility of the production of BBMP was quite poor maybe due to extreme reaction condition and/or trace impurities.\n\n![](images/1f026dbb3269cf62be973a682cb677d7aadc479469083d7119049ce6675b9a06.jpg)  \nScheme 1. Plausible reaction mechanism of BBMP. Ar: 4-[bis(4-methylphenyl)amino]phenyl.\n\nBBMP was found to readily form an amorphous glass. Figure 2 shows DSC curves of BBMP. When the crystalline sample obtained by recrystallization from toluene/hexane was heated, the sample melted at 180 °C (Fig. 2a). When the melt sample was cooled on standing, an amorphous glass formed readily. When the glassy sample was again heated, a glass transition phenomenon was clearly observed at 78 °C (Fig. 2b). The formation of amorphous glass was also confirmed by X-ray diffraction. We could easily obtain the amorphous film of BBMP by spin-coating method onto a transparent glass substrate.\n\n

找出'![](images/1f026dbb3269cf62be973a682cb677d7aadc479469083d7119049ce6675b9a06.jpg)  \nScheme 1. Plausible reaction mechanism of BBMP. Ar: 4-[bis(4-methylphenyl)amino]phenyl.'相关的原文描述

禁用思考模式.
只回答原文内容.

"""

    print(prompt)

    response = requests.post(
        "http://192.168.103.203:11434/api/generate",
        json={
            "model": modelname,
            "prompt": prompt,
            "stream": False
        }
    )
    full_content = response.json()["response"]
    print(full_content)


if __name__ == "__main__":
    unstreamfunc("qwen3:14b","")
