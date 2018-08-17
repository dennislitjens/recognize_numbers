<?php

namespace AppBundle\Controller;

use AppBundle\Entity\mediaEntity;
use Sensio\Bundle\FrameworkExtraBundle\Configuration\Route;
use Symfony\Bundle\FrameworkBundle\Controller\Controller;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\JsonResponse;
use Symfony\Bundle\FrameworkBundle\Console\Application;
use Symfony\Component\Console\Input\ArrayInput;
use Symfony\Component\Console\Output\BufferedOutput;
use Symfony\Component\Process\Process;
use Symfony\Component\Process\Exception\ProcessFailedException;

class DefaultController extends Controller
{
    /**
     * @Route("/", name="homepage")
     */
    public function indexAction(Request $request)
    {
        // replace this example code with whatever you need
        return $this->render('default/index.html.twig', [
            'base_dir' => realpath($this->getParameter('kernel.project_dir')).DIRECTORY_SEPARATOR,
        ]);
    }

    /**
     * @Route("/fileuploadhandler", name="fileuploadhandler")
     */
    public function fileUploadHandler(Request $request) {
        $output = array('uploaded' => false);
        // get the file from the request object
        $file = $request->files->get('file');
        // generate a new filename (safer, better approach), but to use original filename instead, use $fileName = $file->getClientOriginalName();
        $fileName = "number.png";

        // set your uploads directory
        $uploadDir = $this->get('kernel')->getRootDir() . '/../web/uploads/';
        if (!file_exists($uploadDir) && !is_dir($uploadDir)) {
            mkdir($uploadDir, 0775, true);
        }
        if ($file->move($uploadDir, $fileName)) {
            // get entity manager
            $em = $this->getDoctrine()->getManager();

            // create and set this mediaEntity
            $mediaEntity = new mediaEntity();
            $mediaEntity->setFileName($fileName);
        };
        return new JsonResponse("ok");
    }

    /**
     * @Route("/runpython", name="runpython")
     */
    public function runPythonScript(){
        $output = $this->runPythonScriptFromTerminal();

        $uploadDir = $this->get('kernel')->getRootDir() . '/../web/uploads/prepared_image.png';

        return $this->render('default/index.html.twig', [
            'base_dir' => realpath($this->getParameter('kernel.project_dir')).DIRECTORY_SEPARATOR,
            'output' => $output,
            'upload_dir' => $uploadDir
        ]);
    }

    /**
     * @Route("/deletefileresource", name="deleteFileResource")
     */
    public function deleteResource(Request $request){
        $output = array('deleted' => false, 'error' => false);
        $mediaID = $request->get('id');
        $fileName = $request->get('fileName');
        $em = $this->getDoctrine()->getManager();
        $media = $em->find('AppBundle:mediaEntity', $mediaID);
        if ($fileName && $media && $media instanceof mediaEntity) {
            $uploadDir = $this->get('kernel')->getRootDir() . '/../web/uploads/';
            $output['deleted'] = unlink($uploadDir.$fileName);
            if ($output['deleted']) {
                // delete linked mediaEntity
                $em = $this->getDoctrine()->getManager();
                $em->remove($media);
                $em->flush();
            }
        } else {
            $output['error'] = 'Missing/Incorrect Media ID and/or FileName';
        }
        echo $output;
        return new JsonResponse($output);
    }

    public function runPythonScriptFromTerminal(){
        $activateTensorFlowCommand = "source ~/tensorflow/bin/activate";
        $scriptPath = "/Users/dennis/Dropbox/School/Italent/workspace/recignition_keras/gui/web/uploads/predict_cypher_convulotional.py";
        $imagePath = "/Users/dennis/Dropbox/School/Italent/workspace/recignition_keras/gui/web/uploads/number.png";
        $command = $activateTensorFlowCommand.';'.'python '.$scriptPath.' '.$imagePath;
        $process = new Process($command);
        $process->run();

        // executes after the command finishes
        if (!$process->isSuccessful()) {
            throw new ProcessFailedException($process);
        }

        return $process->getOutput();
    }

}
