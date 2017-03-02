import urllib2
import urllib
import time

saveFolder='/home/willie/workspace/TensorFlowWorkshop/data/RedditRoasts/'

nextPageStart='<span class="next-button"><a href="'
nextPageEnd='" rel="nofollow next" >'

imageStart='class="may-blank"><img class="preview" src="'
imageEnd='" width="'

descriptionStart='lang="en" xml:lang="en"><head><title>'
descriptionEnd=' : RoastMe</title><meta name="keywords"'

roastStart='<div class="usertext-body may-blank-within md-container "><div class="md"><p>'
roastEnd='</p>'

roastPostLinkStart='</span><span class="userattrs"></span></p><ul class="flat-list buttons"><li class="first"><a href="'
roastPostLinkEnd='" data-inbound-url="'

numRoasts=1000

requestsMade=0
startRequestTime=0

def getNextPageContents(currentPage):
    nextPageURL=getStringBetween(nextPageStart, nextPageEnd, currentPage)
    print(nextPageURL)
    return getHTML(nextPageURL)
    
def getStringBetween(startStr, endStr, str):
        start=str.index(startStr)+len(startStr)
        end=str.index(endStr)
        return str[start:end]
    
def getStringBetweenStartInd(start, startStr, endStr, str):
        start=str.index(startStr)+len(startStr)
        end=str.index(endStr, start)
        return str[start:end], end

def savePostDescription(currentPage, saveName):
    post=getStringBetween(descriptionStart, descriptionEnd, currentPage)
    saveString(post, saveName)
    
def saveRoasts(currentPage, fileName):
    roasts=[]
    end=0
    start=currentPage.index(roastStart, end)+len(roastStart)
    end=currentPage.index(roastEnd, start)
    while start!=-1:
        roasts+=[currentPage[start:end]]
        try:
            start=currentPage.index(roastStart, end)+len(roastStart)
            end=currentPage.index(roastEnd, start)
        except ValueError:
            start=-1
        
    file=open(fileName, 'w')
    for roast in roasts:
        file.write(roast+'\n')
    file.close()
    
def saveString(string, fileName):
    file=open(fileName, 'w')
    file.write(string)
    file.close()
    
def savePostImage(currentPage, saveName):
    url=getStringBetween(imageStart, imageEnd, currentPage)
    url=url.replace('&amp;', '&')
    urllib.urlretrieve(url, saveName)
    
def getRoastURLs(pageHTML):
    roastPostURLs=[]  
    end=0
    start=pageHTML.index(roastPostLinkStart, end)+len(roastPostLinkStart)
    end=pageHTML.index(roastPostLinkEnd, start)
    while start!=-1:
        roastPostURLs+=[pageHTML[start:end]]
        try:
            start=pageHTML.index(roastPostLinkStart, end)+len(roastPostLinkStart)
            end=pageHTML.index(roastPostLinkEnd, start)
        except ValueError:
            start=-1
    
    return roastPostURLs
    
def getHTML(url):
    global requestsMade
    global startRequestTime
    if requestsMade==58:
        currentTime=int(round(time.time() * 1000))
        while currentTime-startRequestTime<60000:
            currentTime=int(round(time.time() * 1000))
        startRequestTime=int(round(time.time() * 1000))
        requestsMade=0
    url=url.replace('&amp;', '&')
    req=urllib2.Request(url, headers={ 'User-Agent': 'Mozilla/5.0' })
    html=urllib2.urlopen(req).read()
    requestsMade+=1
    return html

#currentPage='https://www.reddit.com/r/RoastMe/?amp%3Bafter=t3_5u11pd&amp%3Bcount=1950'
#currentPage='https://www.reddit.com/r/RoastMe/?amp%3Bafter=t3_5ub0vc&amp%3Bamp%3Bcount=1950&amp%3Bcount=900&count=25&after=t3_5vzqqg'
#currentPage='https://www.reddit.com/r/RoastMe/?amp%3Bafter=t3_5upozs&amp%3Bamp%3Bamp%3Bcount=1950&amp%3Bamp%3Bcount=900&amp%3Bcount=650&count=25&after=t3_5w1p5r'
#currentPage='https://www.reddit.com/r/RoastMe/?amp%3Bafter=t3_5u4x72&amp;amp%3Bamp%3Bamp%3Bamp%3Bcount=1950&amp;amp%3Bamp%3Bamp%3Bcount=900&amp;amp%3Bamp%3Bcount=650&amp;amp%3Bcount=975&amp;count=700&amp;after=t3_5uog84'
currentPage='https://www.reddit.com/r/RoastMe/?amp%3Bafter=t3_5u4x72&amp;amp%3Bamp%3Bamp%3Bamp%3Bcount=1950&amp;amp%3Bamp%3Bamp%3Bcount=900&amp;amp%3Bamp%3Bcount=650&amp;amp%3Bcount=975&amp;count=950&amp;after=t3_5u5d91'

pageHTML=getHTML(currentPage)
startRequestTime=int(round(time.time() * 1000))
requestsMade=0

for roastNum in range(320, numRoasts):
    roastPostURLs=getRoastURLs(pageHTML)
    pageRoastNum=0
    for roastPostURLInd in range(0, len(roastPostURLs)):
        roastPostHTML=getHTML(roastPostURLs[roastPostURLInd])
        try:
            savePostImage(roastPostHTML, saveFolder+'image'+str(roastNum)+'_'+str(pageRoastNum)+'.jpeg')
            savePostDescription(roastPostHTML, saveFolder+'desc'+str(roastNum)+'_'+str(pageRoastNum)+'.txt')
            saveRoasts(roastPostHTML, saveFolder+'roasts'+str(roastNum)+'_'+str(pageRoastNum)+'.txt')
            pageRoastNum+=1
        except ValueError:
            u=0
    pageHTML=getNextPageContents(pageHTML)
    
    
    