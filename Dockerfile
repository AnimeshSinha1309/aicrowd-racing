FROM aicrowd/learn-to-race:base

COPY apt.txt .
RUN apt -qq update && apt -qq install -y `cat apt.txt` \
 && rm -rf /var/cache/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
