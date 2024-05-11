'use client'
import axios from "axios";
import { useState } from "react";


export default function Home() {

  const [newMessages, setNewMessages] = useState('');
  const [messages, setMessages] = useState([{
    sender: 'Model',
    message: 'Hey, how may I help you?'
  }]);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [circularload, setCircularLoad] = useState(false);

  const uploadFile = async () => {
    const formdata = new FormData();
    formdata.append("file", file);

    setCircularLoad(true);

    const upload_file = await axios.post(`http://127.0.0.1:5000/upload`, formdata, {headers : {
      "Content-Type": "multipart/form-data",
    },
  });

  setCircularLoad(false);

  console.log(upload_file);

  }

  const selectFile = (event) => {
    setFile(event.target.files[0]);
  }

  

  const handleNewMessage = (event) => {
    setNewMessages(event.target.value);
  }

  const handleMessage = async (event) => {
    const new_Message = {
      sender: 'User',
      message: newMessages
    };

    messages.push(new_Message);
    // setMessages([...messages, new_Message]);
    const question = newMessages;
    setNewMessages('');

    setLoading(true);

    const response = await axios.get(`http://127.0.0.1:5000/ask?query=${question}`);
    console.log(response);
    setNewMessages('');

    setLoading(false);

    messages.push({sender:'Model', message: response.data.answer.result});
    

    console.log(messages);
  }


  return (
    <main className="flex justify-center min-h-screen p-25">
      <div className="w-full font-mono text-sm lg:flex flex-col space-y-8 ">
        
        <div className="flex flex-col space-y-4 mx-10 my-10">
          {
            messages.map((m, index) => (
              <div key={index} className={m.sender=='Model'? "chat chat-start" : "chat chat-end"}>
                <div className="chat-bubble">{m.message}</div>
              </div>
            ))
          }

          <span className={loading? "loading loading-bars loading-md" : "hidden"}></span>

    
        </div>



        <div className="flex items-center justify-around fixed bottom-10 w-full">

          <input onChange={selectFile} type="file" className="file-input file-input-bordered file-input-accent mx-5 min-w-80" />
          
          <button onClick={uploadFile} className={circularload ? "hidden" : "btn btn-outline btn-accent"}> Upload </button>

          <span className={circularload ? "loading loading-spinner text-accent size-8" : "hidden" }></span>

          <input onChange={handleNewMessage} value={newMessages} type="text" placeholder="Type here" className="input input-bordered w-full mx-5" />

          <button onClick={handleMessage} className="btn btn-outline btn-accent mx-5 px-8"> ASK </button>

        </div>

      </div>
    </main>
  );
}


