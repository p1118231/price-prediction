{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "55398d3f-c1d3-4e4e-b0c1-8bd172ac5226",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:50663\n",
      "Press CTRL+C to quit\n",
      " * Restarting with watchdog (fsevents)\n",
      "0.00s - Debugger warning: It seems that frozen modules are being used, which may\n",
      "0.00s - make the debugger miss breakpoints. Please pass -Xfrozen_modules=off\n",
      "0.00s - to python to disable frozen modules.\n",
      "0.00s - Note: Debugging will proceed. Set PYDEVD_DISABLE_FILE_VALIDATION=1 to disable this validation.\n",
      "Traceback (most recent call last):\n",
      "  File \"<frozen runpy>\", line 198, in _run_module_as_main\n",
      "  File \"<frozen runpy>\", line 88, in _run_code\n",
      "  File \"/opt/anaconda3/lib/python3.12/site-packages/ipykernel_launcher.py\", line 17, in <module>\n",
      "    app.launch_new_instance()\n",
      "  File \"/opt/anaconda3/lib/python3.12/site-packages/traitlets/config/application.py\", line 1074, in launch_instance\n",
      "    app.initialize(argv)\n",
      "  File \"/opt/anaconda3/lib/python3.12/site-packages/traitlets/config/application.py\", line 118, in inner\n",
      "    return method(app, *args, **kwargs)\n",
      "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"/opt/anaconda3/lib/python3.12/site-packages/ipykernel/kernelapp.py\", line 654, in initialize\n",
      "    self.init_sockets()\n",
      "  File \"/opt/anaconda3/lib/python3.12/site-packages/ipykernel/kernelapp.py\", line 331, in init_sockets\n",
      "    self.shell_port = self._bind_socket(self.shell_socket, self.shell_port)\n",
      "                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"/opt/anaconda3/lib/python3.12/site-packages/ipykernel/kernelapp.py\", line 253, in _bind_socket\n",
      "    return self._try_bind_socket(s, port)\n",
      "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"/opt/anaconda3/lib/python3.12/site-packages/ipykernel/kernelapp.py\", line 229, in _try_bind_socket\n",
      "    s.bind(\"tcp://%s:%i\" % (self.ip, port))\n",
      "  File \"/opt/anaconda3/lib/python3.12/site-packages/zmq/sugar/socket.py\", line 302, in bind\n",
      "    super().bind(addr)\n",
      "  File \"zmq/backend/cython/socket.pyx\", line 564, in zmq.backend.cython.socket.Socket.bind\n",
      "  File \"zmq/backend/cython/checkrc.pxd\", line 28, in zmq.backend.cython.checkrc._check_rc\n",
      "zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:50637')\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[0;31mSystemExit\u001b[0m\u001b[0;31m:\u001b[0m 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/anaconda3/lib/python3.12/site-packages/IPython/core/interactiveshell.py:3585: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "from prophet import Prophet\n",
    "import socket\n",
    "from prophet.serialize import model_from_json\n",
    "import json\n",
    "\n",
    "app = Flask(__name__)\n",
    "# Load models\n",
    "# Method 1: Using fin.read() consistently across all models\n",
    "with open('daily_model.json', 'r') as fin:\n",
    "    model_daily = model_from_json(fin.read())  \n",
    "\n",
    "with open('weekly_model.json', 'r') as fin:\n",
    "    model_weekly = model_from_json(fin.read())  # Changed from json.load()\n",
    "\n",
    "with open('monthly_model.json', 'r') as fin:\n",
    "    model_monthly = model_from_json(fin.read())  # Changed from json.load()\n",
    "\n",
    "\n",
    "def find_free_port():\n",
    "    \"\"\"Finds a free port by binding to an ephemeral port assigned by the OS.\"\"\"\n",
    "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n",
    "    s.bind(('', 0))  # Bind to a free port provided by the host.\n",
    "    free_port = s.getsockname()[1]\n",
    "    s.close()\n",
    "    return free_port\n",
    "\n",
    "\n",
    "@app.route('/predict/all', methods=['GET'])\n",
    "def predict_all():\n",
    "    today = pd.to_datetime('today')\n",
    "    dates_daily = pd.date_range(start=today, periods=1, freq='D').to_frame(index=False, name='ds')\n",
    "    dates_weekly = pd.date_range(start=today, periods=7, freq='D').to_frame(index=False, name='ds')\n",
    "    dates_monthly = pd.date_range(start=today, periods=30, freq='D').to_frame(index=False, name='ds')\n",
    "\n",
    "    forecast_daily = model_daily.predict(dates_daily)\n",
    "    forecast_weekly = model_weekly.predict(dates_weekly)\n",
    "    forecast_monthly = model_monthly.predict(dates_monthly)\n",
    "\n",
    "    return jsonify({\n",
    "        'daily_sales': float(forecast_daily['yhat'].sum()),\n",
    "        'weekly_sales': float(forecast_weekly['yhat'].sum()),\n",
    "        'monthly_sales': float(forecast_monthly['yhat'].sum())\n",
    "    })\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    port = find_free_port()\n",
    "    app.run(port= port,debug=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d439465e-aeb5-4f6e-8c98-389b754c0d17",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
