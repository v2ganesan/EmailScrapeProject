def save_open_attachment(service, attachment, message_id, save_path):
   attachment_id = attachment['attachmentId']
   attach_data = service.users().messages().attachments().get(userId='me', messageId=message_id, id=attachment_id).execute()
   data = base64.urlsafe_b64decode(attach_data['data'].encode('UTF-8'))

   if not os.path.exists(save_path):
      os.makedirs(save_path)
  
   file_path = os.path.join(save_path, attachment['filename'])
   with open(file_path, 'wb') as f:
      f.write(data)
   print(f"Attachment saved to {file_path}")

   if os.name =='posix':
      os.system(f'open {file_path}')
   elif os.name == 'nt':
      os.startfile(file_path)


def resize_png(png_path, output_size=244):
   #Open the image
   image = Image.open(png_path).convert("RGB")

   #Calculate the resize 
   aspect_ratio = image.width / image.height
   if aspect_ratio > 1:
      new_width = output_size
      new_height = int (output_size / 2)
   else:
      new_height = output_size
      new_width = int(output_size / 2)

   #resize the image 
   image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

   #create a new image and paste the previous image 
   new_image = Image.new("RGB", (output_size, output_size), (255, 255, 255))
   paste_position = ((output_size - new_width) // 2, (output_size - new_height) // 2)
   new_image.paste(image, paste_position)
   
   image.save(png_path)
   # Open the HTML file with the default web browser
   if os.name == 'posix':  # For macOS and Linux
        os.system(f'open "{png_path}"')
   elif os.name == 'nt':  # For Windows
        os.startfile(screenshot_path)
        
def take_screenshot_of_file(html_path, screenshot_path):
   #set up the web driver
   options = webdriver.ChromeOptions()
   options.add_argument("--headless")
   driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

   driver.get(f"file://{html_path}")

   #find the height of the page
   # Calculate the total height of the page
   total_height = driver.execute_script("return document.body.scrollHeight")
   total_width = driver.execute_script("return document.body.scrollWidth")
   #num_scrolls = total_height // viewport_height

   driver.set_window_size(total_width, total_height)

   screenshot_image = driver.get_screenshot_as_png()
   driver.quit()

   screenshot = Image.open(BytesIO(screenshot_image))
   screenshot.save(screenshot_path)

    # Open the HTML file with the default web browser
   if os.name == 'posix':  # For macOS and Linux
        os.system(f'open "{screenshot_path}"')
   elif os.name == 'nt':  # For Windows
        os.startfile(screenshot_path)
   
   resize_png(screenshot_path)