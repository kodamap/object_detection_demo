$(function () {
    var detection_cmd = ['async', 'sync', 'object_detection',
        'age_gender_detection', 'face_detection',
        'emotions_detection', 'head_pose_detection',
        'facial_landmarks_detection'];
    var flip_cmd = ['flip'];
    var url = "";
    $('.btn').on('click', function () {
        var command = JSON.stringify({ "command": $('#' + $(this).attr('id')).val() });
        if (JSON.parse(command).command == "") {
            var command = JSON.stringify({ "command": $(this).find('input').val() });
        }
        //console.log(command)
        if (detection_cmd.includes(JSON.parse(command).command)) {
            url = '/detection';
            post(url, command);
        }
        if (flip_cmd.includes(JSON.parse(command).command)) {
            url = '/flip';
            post(url, command);
        }
    });
    function post(url, command) {
        $.ajax({
            type: 'POST',
            url: url,
            data: command,
            contentType: 'application/json',
            timeout: 10000
        }).done(function (data) {
            var sent_cmd = JSON.parse(command).command;
            var is_async_mode = JSON.parse(data.ResultSet).is_async_mode;
            var flip_code = JSON.parse(data.ResultSet).flip_code;
            var is_obj_det = JSON.parse(data.ResultSet).is_object_detection;
            var is_face_det = JSON.parse(data.ResultSet).is_face_detection;
            var is_ag_det = JSON.parse(data.ResultSet).is_age_gender_detection;
            var is_em_det = JSON.parse(data.ResultSet).is_emotions_detection;
            var is_hp_det = JSON.parse(data.ResultSet).is_head_pose_detection;
            var is_lm_det = JSON.parse(data.ResultSet).is_facial_landmarks_detection;
            //console.log(sent_cmd);
            $("#res").text("obj:" + is_obj_det + " face:" + is_face_det + " ag:" + is_ag_det + " em:" + is_em_det + " hp:" + is_hp_det + " lm:" + is_lm_det);
            if (sent_cmd == 'async') {
                $("#async").attr('class', 'btn btn-danger');
                $("#sync").attr('class', 'btn btn-dark');
            }
            if (sent_cmd == 'sync') {
                $("#sync").attr('class', 'btn btn-danger');
                $("#async").attr('class', 'btn btn-dark');
            }
            if (sent_cmd == 'object_detection') {
                $("#is_face_detection").attr("disabled", true);
            }
            if (sent_cmd == 'face_detection') {
                $("#is_face_detection").attr("disabled", false);
            }
        }).fail(function (jqXHR, textStatus, errorThrown) {
            $("#res").text(textStatus + ":" + jqXHR.status + " " + errorThrown);
        });
        return false;
    }
    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    });
});

